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
from peft.tuners.adaption_prompt.config import AdaptionPromptConfig, TRANSFORMERS_MODEL_CONFIG
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


sep='</s>'
eos='</s>'
def prepare_inputs_for_generation(
        input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
    position_ids = kwargs.get("position_ids", None)
    if past_key_values is not None:
        # print(past_key_values[0][0].shape[-2], end='')
        # print(input_ids.shape[1], end=' ')
        position_ids = torch.ones([input_ids.shape[0],1], dtype=torch.long)*(input_ids.shape[1]-1)

        new_key_value=[]
        for i in range(len(past_key_values)):
            layer_key_value=[]
            for j in range(len(past_key_values[i])):
                layer_key_value.append(past_key_values[i][j].expand([input_ids.shape[0],-1,-1,-1]))#(B, 40, n*k, 128)
            new_key_value.append(torch.stack(layer_key_value))
        past_key_values = torch.stack(new_key_value)
        input_ids = input_ids[:, -1:]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

class EncoderAdaptedAttention(peft.tuners.adaption_prompt.AdaptedAttention):
    
    def __init__(self, adapter_layer_idx:int, *args, **kargs):
        super().__init__( *args, **kargs,)
        del self.adaption_prompt
        self.adapter_layer_idx = adapter_layer_idx
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
        if kwargs.get("adaption_prompt", False):
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

        # (bsz, num_key_value_heads, adapter_len, head_dim)
        adapter_k = (
            key.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        adapter_v = (
            value.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
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
    #overide
    def forward(self, *args, prefix = None, **kargs):
        print(prefix)
            


        output = self.model(*args, **kargs)
        return output
    def __call__(self, *args, prefix = None, **kargs):
        return self.forward(*args, prefix=prefix, **kargs)

        
class LLaMa_reader(torch.nn.Module):
    def __init__(self, model_dir, device, adapter_layers=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token='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')
        self.tokenizer.model_max_length=2048
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.eos_id

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

        peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=adapter_layers, adapter_len=1)}
        self.model = peft.AdaptionPromptModel(self.model, configs = peft_configs, adapter_name='Enc')
        self.model.training=False
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None

    def forward(self, *args, **kwargs)->tuple[Tensor, Tensor]:
        '''
        forward function for teacher forcing\\
        the shape of ids, target, masks is (B,n)\\
        the shape of encoder_output is (B, 40, 2, 40, nd, 128), encoder_masks is (B,nd)\\
        '''


        labels = kwargs.get('labels', None)
        if labels is not None:
            del kwargs['labels']


        
        if kwargs.get('prefix', False):
            for par, p in zip(self.model._parents['Enc'], kwargs['prefix']):
                setattr(par, "adaption_prompt", p)
            del kwargs['prefix']


        output = self.model(**kwargs)


        if kwargs.get('prefix', False):
            for par in self.model._parents['Enc']:
                delattr(par, "adaption_prompt")
        lm_logits:Tensor = output.logits
        labels:Tensor
        
        if labels is not None:
            labels = labels
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = -torch.log_softmax(shift_logits, dim=-1)[torch.arange(shift_labels.shape[0])[:,None], torch.arange(shift_labels.shape[1])[None,:], shift_labels] #(B,N)
            mask = shift_labels==-100
            loss = loss.masked_fill_(mask, 0)
            loss = loss.sum(-1)/(~mask).sum(-1)
        
            return lm_logits, loss
        return lm_logits

    @torch.inference_mode()
    def generate(self, message, encoder_output = None, encoder_masks=None, max_new_tokens = 1024, test=False, streamer=True):
        '''
        for inference, batch size is one.
        '''
        #tokenize
        if streamer:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer=None
        tokens = self.tokenizer(message, padding=True ,truncation=True,return_tensors='pt')
        ids=tokens.input_ids.to(self.model.device)
        masks=tokens.attention_mask.to(self.model.device)
        message_len=ids.shape[1]

        if test:#for testing encoder output
            print(tokens)
            B=1
            n=10
            
            num_heads = self.model.config.num_key_value_heads
            num_layers = self.model.config.num_hidden_layers
            num_dims = self.model.config.hidden_size//num_heads
            encoder_output=torch.zeros([num_layers,2,B,num_heads,n,num_dims], dtype=torch.float16).to(self.model.device)
            encoder_masks = torch.zeros([B,n], dtype=torch.long).to(self.model.device)

        #concat document mask
        if encoder_output is not None:
            attention_mask = torch.cat([encoder_masks, masks],dim=-1)
            position_ids = torch.arange(ids.shape[1]).tile([ids.shape[0],1])
            # position_ids=None

            #need forward first because generate will only use last ids
            output=self.model.forward(input_ids = ids[:,:-1],
                                        attention_mask=attention_mask[:,:-1],
                                        position_ids = position_ids[:,:-1],
                                        past_key_values = encoder_output,
                                        use_cache=True )

            past_key_values = output.past_key_values
        else:
            attention_mask = masks
            past_key_values = None
            position_ids=None


        generate_ids = self.model.generate(input_ids = ids,
                                           attention_mask = attention_mask,
                                           past_key_values = past_key_values,
                                           use_cache=True,
                                           max_new_tokens=max_new_tokens,
                                           streamer=streamer,
                                           **self.generate_config)

        output = self.tokenizer.decode(generate_ids[0][ids.shape[1]:],skip_special_tokens=True)
        self.chat_history.append([message, output])
        return output


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
