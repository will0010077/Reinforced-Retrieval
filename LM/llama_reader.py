from typing import Dict, List
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer,AutoModelForCausalLM
# from auto_gptq import AutoGPTQForCausalLM
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import peft
from peft.tuners.adaption_prompt.config import AdaptionPromptConfig, TRANSFORMERS_MODEL_CONFIG, prepare_config
import math


with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


sep='</s>'
eos='</s>'

class EncoderAdaptedAttention(peft.tuners.adaption_prompt.AdaptedAttention):
    
    def __init__(self, *args, **kargs):
        super().__init__( *args, **kargs,)
        del self.adaption_prompt
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
        scores = self.adaption_gate.tanh() * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
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

class EncoderAdaptedModel(peft.AdaptionPromptModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
        """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
        for par in parents:
            attn = EncoderAdaptedAttention(
                model_type=self.model.config.model_type,
                adapter_len=config.adapter_len,
                model=getattr(par, config.target_modules),
            )
            setattr(par, config.target_modules, attn)
        
class LLaMa_reader(torch.nn.Module):
    def __init__(self, model_dir, device, adapter_layers=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token='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')
        self.tokenizer.model_max_length=2048
        self.tokenizer.padding_side='left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token

        self.generate_config=config['generate_config']
        # self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
        #     trust_remote_code=False,
        #     device_map="auto",
        #     # use_triton=False,
        #     # use_cache=True,
        #     # trainable=True,
        #     )
        self.model=AutoModelForCausalLM.from_pretrained(model_dir, token='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', device_map=device, use_cache=True, torch_dtype=torch.float16)
        # print(self.model)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.stop_strings=[]
        self.model.train(False)
        self.model.requires_grad_(False)

        peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=adapter_layers, adapter_len=1)}
        self.model = EncoderAdaptedModel(self.model, configs = peft_configs, adapter_name='Enc')
        self.module_name = prepare_config(peft_configs['Enc'], self.model).target_modules
        # for par in self.model._parents['Enc']:
        #     getattr(par, self.module_name).adaption_gate.requires_grad_(True)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None

    def forward(self, *args, prefix=None, **kwargs)->tuple[Tensor, Tensor]:
        '''
        forward function for teacher forcing\\
        the shape of ids, target, masks is (B,n)\\
        the shape of encoder_output is (B, 40, 2, 40, nd, 128), encoder_masks is (B,nd)\\
        '''


        labels = kwargs.get('labels', None)
        if labels is not None:
            del kwargs['labels']


        self._set_prefix(prefix)
        output = self.model(**kwargs)
        self._del_prefix(prefix)
        lm_logits:Tensor = output.logits
        labels:Tensor
        del output.past_key_values
        if labels is not None:
            labels = labels
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = -torch.log_softmax(shift_logits, dim=-1)[torch.arange(shift_labels.shape[0])[:,None], torch.arange(shift_labels.shape[1])[None,:], shift_labels] #(B,N)
            mask = shift_labels==-100
            loss = loss.masked_fill_(mask, 0)
            output.loss = loss.sum(-1)/(~mask).sum(-1)

        
        return output

    @torch.inference_mode()
    def generate(self, message, cache = None,  max_new_tokens = 1024, streamer=False, prefix=None, stop_strings=[]):
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
        
        self._set_prefix(prefix)
        self.model.generation_config.stop_strings.extend(stop_strings)
        outputs = self.model.generate(**tokens,
                                        streamer=streamer,
                                        max_new_tokens=max_new_tokens,
                                        past_key_values = cache,
                                        **self.generate_config)
        
        [self.model.generation_config.stop_strings.pop(-1) for _ in range(len(stop_strings))]
        self._del_prefix(prefix)
        output = self.tokenizer.batch_decode(outputs[:,tokens.input_ids.shape[1]:],skip_special_tokens=True)
        self.chat_history.append([message, output])
        return output
    def _set_prefix(self, prefix):
        
        if prefix is not None:
            for par, p in zip(self.model._parents['Enc'], prefix):
                setattr(getattr(par, self.module_name), "adaption_prompt", p)
    def _del_prefix(self, prefix):
        if prefix is not None:
            for par in self.model._parents['Enc']:
                delattr(getattr(par, self.module_name), "adaption_prompt")

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
