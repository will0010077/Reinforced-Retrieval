
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer,AutoModelForCausalLM
# from auto_gptq import AutoGPTQForCausalLM
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
class LLaMa_reader:
    def __init__(self, model_dir):
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
        self.model=AutoModelForCausalLM.from_pretrained(model_dir, token='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', device_map='cpu', torch_dtype=torch.float16)
        # print(self.model)
        self.model.to(device)
        self.model.prepare_inputs_for_generation = prepare_inputs_for_generation
        self.model.training=True
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None

    def forward(self, ids, target=None, masks=None, encoder_output = None, encoder_masks=None)->tuple[Tensor, Tensor]:
        '''
        forward function for teacher forcing\\
        the shape of ids, target, masks is (B,n)\\
        the shape of encoder_output is (B, 40, 2, 40, nd, 128), encoder_masks is (B,nd)\\
        '''


        #concat document mask
        if encoder_masks is not None:
            if masks is None:
                masks = torch.ones_like(ids, dtype = torch.long, device=encoder_masks.device)
            attention_mask = torch.cat([encoder_masks, masks.to(encoder_masks.device)],dim=-1)
        else:
            attention_mask = None
        position_ids = torch.arange(ids.shape[1]).tile([ids.shape[0],1])
        output = self.model(input_ids=ids,
                            position_ids = position_ids,
                            attention_mask=attention_mask,
                            labels=target,
                            past_key_values=encoder_output,
                            use_cache=True)

        return output.logits, output.loss

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
