from typing import Dict, List
from transformers import TextStreamer,AutoTokenizer,AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, GenerationConfig

# from auto_gptq import AutoGPTQForCausalLM
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from tqdm import tqdm
import yaml
import peft
from peft.utils import _freeze_adapter, _get_submodules
from LM.Knowledge_encoder import KnowEncoder
from peft.tuners.adaption_prompt.config import AdaptionPromptConfig, TRANSFORMERS_MODEL_CONFIG, prepare_config
import math
from config import generate_config

sep='</s>'
eos='</s>'

class EncoderAdaptedAttention(peft.tuners.adaption_prompt.AdaptedAttention):
    
    def __init__(self, *args, **kargs):
        super().__init__( *args, **kargs,)
        del self.adaption_prompt
        self.adaption_gate.data = self.adaption_gate.data.float() # change it to float32 to avoid NaN
    # overide
    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")
        output, _, past_key_value = self.model(**kwargs)
        if not hasattr(self, "adaption_prompt"):
            return output, None, past_key_value
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        factor = (
            self.model.k_proj.in_features // self.model.k_proj.out_features
        )  # Mistral has different input and output dimension for k_proj and v_proj layers


        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)

        adapter_len = self.adaption_prompt.shape[-2]
        # (bsz, num_key_value_heads, adapter_len, head_dim)
        adapter_k = (
            key.view(-1, adapter_len, (self.model.num_heads // factor), self.model.head_dim)
            .repeat(bsz//key.shape[0], 1, 1, 1)
            .transpose(1, 2)
        )
        adapter_v = (
            value.view(-1, adapter_len, (self.model.num_heads // factor), self.model.head_dim)
            .repeat(bsz//key.shape[0], 1, 1, 1)
            .transpose(1, 2)
        )
        # Below is taken from https://github.com/huggingface/transformers/blob/e547458c43dfdbbb8f6a7757237e234c44e20a8f/src/transformers/models/mistral/modeling_mistral.py#L181
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)
        # Recompute query states.
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)

        previous_dtype = query_states.dtype

        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(
            self.model.head_dim
        )
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = (self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32)).to(previous_dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original output.
        output = output + adapter_output

        # Restore original dtype.
        output = output.to(previous_dtype)
        return output, None, past_key_value

class LLaMa_reader(torch.nn.Module):
    def __init__(self, model_dir, device, token, from_pretrained=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token=token)
        self.tokenizer.model_max_length=2048
        self.tokenizer.padding_side='left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token

        self.generate_config=generate_config
        if from_pretrained:
            self.model=AutoModelForCausalLM.from_pretrained(model_dir, token=token, device_map=device, use_cache=True, torch_dtype = torch.bfloat16)
        else:
            LM_config = LlamaConfig.from_pretrained(model_dir, token=token, device_map=device)
            self.model=LlamaForCausalLM(LM_config).to(LM_config.torch_dtype)
            self.model.generation_config = GenerationConfig.from_pretrained(model_dir)
        self.dtype = self.model.config.torch_dtype
        # print(self.model)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.stop_strings=[]
        self.model.train(True)
        self.model.requires_grad_(False)
        self.config = self.model.config
        # for par in self.model._parents['Enc']:
        #     getattr(par, self.module_name).adaption_gate.requires_grad_(True)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None
    @property
    def device(self):
        return self.model.device
    def forward(self, tokens, return_logits = False)->tuple[Tensor, Tensor]:
        '''
        forward function for teacher forcing\\
        the shape of tokens is (B,n)\\
        output: lm_logits(B,n), loss(B)
        '''

        # print(self.model.model.model.embed_tokens.weight.device, tokens.input_ids.device, prefix[0].device)
        
        output = self.model(**tokens)
        lm_logits:Tensor = output.logits
        labels:Tensor
        if not return_logits:
            del output.logits
        del output.past_key_values
        loss=None

        labels = tokens.get('labels', None)
        if labels is not None:
            del tokens['labels']
            # Shift so that tokens < n predict n

            logp = torch.log_softmax(lm_logits, dim=-1)
            shift_logp = logp[..., :-1, :]
            labels[tokens['attention_mask']==0]=-100
            shift_labels = labels[..., 1:]
            
            loss = -shift_logp[torch.arange(shift_labels.shape[0])[:,None], torch.arange(shift_labels.shape[1])[None,:], shift_labels] #(B,N)
            mask = shift_labels==-100
            loss = loss.masked_fill_(mask, 0)
            
            loss = loss.sum(-1)/(~mask).sum(-1)

        
        return lm_logits, loss

    @torch.inference_mode()
    def generate(self, message, cache = None,  max_new_tokens = 1024, streamer=False, stop_strings=[], decode = True):
        '''
        for inference, batch size is one.
        '''
        #tokenize
        if streamer:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = None
        tokens = self.tokenizer(message, padding=True ,truncation=True,return_tensors='pt')
        tokens = tokens.to(self.model.device)
        
        self.model.generation_config.stop_strings.extend(stop_strings)
        outputs = self.model.generate(**tokens,
                                        streamer=streamer,
                                        max_new_tokens=max_new_tokens,
                                        past_key_values = cache,
                                        **self.generate_config)
        
        [self.model.generation_config.stop_strings.pop(-1) for _ in range(len(stop_strings))]

        outputs = [outputs[j][len(tokens.input_ids[0]):].cpu() for j in range(len(tokens.input_ids))]
        if decode:
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # self.chat_history.append([message, output])[:,tokens.input_ids.shape[1]:]
        return outputs
    
    
    @torch.inference_mode()
    def pseudo_generate(self, messages:list[str], forcing:list[str], temperture = 1, return_prob = False, decode = True,  **kwargs):
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(forcing, str):
            forcing = [forcing]
        cat = [m+" "+f+self.eos for m, f in zip(messages, forcing)]
        unlabel = self.tokenizer(text=messages).input_ids
        unlabel_len = [len(m) for m in unlabel]
        # print(max([len(s) for s in unlabel]))
        tokens = self.tokenizer(text=cat, return_tensors='pt', padding=True, max_length=1024, truncation =True,)
        tokens = tokens.to(self.model.device)
        
        lm_logits, loss = self.forward(tokens, return_logits = True, **kwargs)
        dist = Categorical(logits=torch.log_softmax(lm_logits/temperture, dim=-1))
        
        top_token = dist.sample()
        cut_token = [top_token[i][tokens.attention_mask[i].bool()][unlabel_len[i]-1:-1] for i in range(len(messages))]
        if decode:
            output = self.tokenizer.batch_decode(cut_token, skip_special_tokens=False)
        else:
            output = cut_token
        if return_prob:
            log_prob = dist.log_prob(top_token)
            token_prob = [log_prob[i][tokens.attention_mask[i].bool()][unlabel_len[i]-1:-1].cpu() for i in range(len(messages))]
            return output, token_prob
        return output
        

class EncTunedLM(peft.AdaptionPromptModel, nn.Module):

    def __init__(self, LM:LLaMa_reader, Enc:KnowEncoder, configs: Dict, adapter_name: str):
        
        nn.Module.__init__(self)
        self.model = LM
        self.Enc = Enc
        # Store adapter configs by name.
        self.peft_config: Dict[str, AdaptionPromptConfig] = {}
        # Store lists of the parents of the affected attention modules by adapter name.
        # We keep references to the parents so we can swap the adapters in-and-out of the model.
        self._parents: nn.ParameterDict[str, List[nn.Module]] = nn.ParameterDict()
        # Store lists of cached AdaptedAttention modules by name.
        self._cached_adapters: Dict[str, List] = {}
        # The name of the currently active adapter.
        self._active_adapter = None
        # Whether the adapter is enabled.
        self._enabled = True
        # self.forward = self.model.forward # Fuck this!!!
        self.add_adapter(adapter_name, configs[adapter_name])
        self._mark_only_adaption_prompts_as_trainable(self.model)
        self.module_name = prepare_config(peft.AdaptionPromptConfig, self.model).target_modules

    def forward(self, *args, Doc_tokens = None, k = 1, use_ref = False, **kwargs):

        if Doc_tokens is not None:
            prefix = self.Enc.forward(Doc_tokens)
        else:
            prefix = None
        self._set_prefix(prefix)
        output = self.model.forward(*args, **kwargs)
        self._del_prefix(prefix)

        ref_logp=None
        if use_ref:
            with torch.no_grad():
                ref_logp, loss = self.model.forward(*args, **kwargs)
                ref_logp = ref_logp.to(torch.bfloat16)

        return ref_logp, output
    
    @torch.inference_mode()
    def pseudo_generate(self, messages:list[str], forcing:list[str], Doc_tokens = None, **kwargs):
        
        if Doc_tokens is not None:
            prefix = self.Enc.forward(Doc_tokens)
        else:
            prefix = None
        self._set_prefix(prefix)
        output = self.model.pseudo_generate(messages, forcing, **kwargs)
        self._del_prefix(prefix)
        return output
    
    def generate(self, messages, Doc_tokens = None, **kwargs):
        
        if Doc_tokens is not None:
            prefix = self.Enc.forward(Doc_tokens)
        else:
            prefix = None
        self._set_prefix(prefix)
        output = self.model.generate(messages, **kwargs)
        self._del_prefix(prefix)
        return output
    
    def add_adapter(self, adapter_name: str, config: AdaptionPromptConfig) -> None:
        """Add an adapter with the given name and config."""
        config = prepare_config(config, self.model)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")

        parents = nn.ParameterList()
        for name, _ in self.model.named_modules():
            if name.endswith(config.target_modules):
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
        if len(parents) < config.adapter_layers:
            raise ValueError(
                f"Config specifies more adapter layers '{config.adapter_layers}'"
                f" than the model has '{len(parents)}'."
            )
        # Note that if the target modules are not in Sequential, ModuleList, or
        # some other PyTorch ordered container, the behavior is undefined as we
        # assume here that the order of the modules is the same as the order of
        # the transformer decoder layers.
        parents = parents[-config.adapter_layers :]
        self._parents[adapter_name] = parents

        # It is only None during initialization.
        # If it is disabled, we don't have to remove the modules.
        if self._active_adapter is not None and self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        self._active_adapter = adapter_name
        self.peft_config[adapter_name] = config
        self._create_adapted_attentions(config, parents)
        if not self._enabled:
            self._remove_adapted_attentions(self._active_adapter)

        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
        """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
        for par in parents:
            attn = EncoderAdaptedAttention(
                model_type=self.model.config.model_type,
                adapter_len=config.adapter_len,
                model=getattr(par, config.target_modules),
            )
            setattr(par, config.target_modules, attn)
        
    def _set_prefix(self, prefix):
        
        if prefix is not None:
            for par, p in zip(self._parents['Enc'], prefix):
                setattr(getattr(par, self.module_name), "adaption_prompt", p)

    def _del_prefix(self, prefix):
        if prefix is not None:
            for par in self._parents['Enc']:
                delattr(getattr(par, self.module_name), "adaption_prompt")
'''We have 40 pounds of product ready to ship, ready to go.

Are you ready?

Who the hell are you?

You know.

You all know exactly who I am.

Say my name.

You what?

I don't have a damn clue who the hell you are.

Yeah, you do.

I'm the cook.

I'm the man who killed Gus Fring.

Bullshit.

Cartel got Fring.

You sure?

That's right.

Now, say my name.

Heisenberg.

You're goddamn right.'''

if __name__=="__main__":
    inferencer = LLaMa_reader("TheBloke/Llama-2-13B-chat-GPTQ")#TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")

    inferencer.system_prompt="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and as short as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    #"A chat between user and an assistant called llama. The assistant base on history gives helpful, detailed, and polite answers. Use 繁體中文\n. Do not repeat user question."
    def response_custom(s:str):
        ids=[]
        old_string=''
        for i in inferencer.generator(s):
            ids+=[i]
            new_string = inferencer.tokenizer.decode(ids, skip_special_tokens=True)
            print(new_string[-(len(new_string)-len(old_string)):], end='',flush=True)
            old_string=new_string

        return inferencer.generator(s)
    
    def response(s:str):
        return inferencer.generate(s, test=True)

    while True:
        s = input("User: ")
        if s != '':
            # inferencer.generate(s)
            print(response(s))
