
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
sep='</s>'
eos='</s>'
def prepare_inputs_for_generation(
        input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False)
        self.tokenizer.model_max_length=2048
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.eos_id

        self.generate_config=dict(
                            max_new_tokens = 1024,
                            no_repeat_ngram_size = 20,
                            do_sample  = False,
                            num_beams = 1,
                            bad_words_ids = None
                            )
        self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            use_cache=True,
            strict=False)
        self.model.model.prepare_inputs_for_generation = prepare_inputs_for_generation
        print(self.model)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.thread_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def forward(self, ids, target=None, masks=None, encoder_output = None, encoder_masks=None):
        '''
        forward function for teacher forcing
        the shape of ids & target & masks is (B,n)
        the shape of encoder_output is (B, 40, 2, 40, nd, 128), encoder_masks is (B,nd)
        '''

        output = self.model(input_ids=ids,
                            position_ids = torch.arange(ids.shape[1]).tile([ids.shape[0],1]),
                            attention_mask=torch.cat([encoder_masks, masks],dim=-1),
                            labels=target,
                            past_key_values=encoder_output,
                            use_cache=True)

        return output.logits, output.loss

    def generator(self, message, encoder_output = None, encoder_masks=None, max_new_tokens = 1024, test=True):
        '''
        for inference
        '''
        tokens = self.tokenizer(message, padding=True ,truncation=True,return_tensors='pt')
        ids=tokens.input_ids.to(self.model.device)
        masks=tokens.attention_mask.to(self.model.device)
        message_len=ids.shape[1]
        if test:
            print(tokens)
            B=1
            n=10
            encoder_output=torch.zeros([40,2,B,40,n,128], dtype=torch.float16).to(self.model.device)
            encoder_masks = torch.zeros([B,n], dtype=torch.long).to(self.model.device)

        for n in range(max_new_tokens):
            if n>0:
                ids = torch.tensor([[generate_ids]],device=self.model.device)
                masks = torch.cat([masks, torch.ones(masks.shape[0],1, dtype=torch.long, device=self.model.device)], dim=1)
                encoder_output = output.past_key_values

            # tokens_len=encoder_output[0][0].shape[-2]
            position_ids=torch.arange(n-ids.shape[1]+message_len, n+message_len).tile([ids.shape[0],1])
            # position_ids = None
            attention_mask=torch.cat([encoder_masks, masks],dim=-1) if encoder_masks is not None else None
            output=self.model.forward(input_ids = ids,
                                      attention_mask=attention_mask,
                                      position_ids = position_ids,
                                      past_key_values=encoder_output,
                                      use_cache=True )


            generate_ids = torch.argmax(output.logits[0,-1], dim=0)

            # for i in range(len(output.past_key_values)):
            #     for j in range(len(output.past_key_values[i])):
            #         print(f'({i}, {j})',output.past_key_values[i][j].shape)#(1,40,2,128)->(B,head,n,dim)


            yield generate_ids
            if generate_ids == self.eos_id:
                break
    def generate(self, message: str, encoder_output = None):
        tokens = self.tokenizer(message, return_tensors='pt').input_ids
        attention_mask = torch.ones([1, tokens.shape[1]])
        generate_ids = self.model.generate(input_ids=tokens.cuda(), past_key_values=encoder_output, attention_mask = attention_mask,streamer=self.streamer, **self.generate_config)

        output = self.tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        self.chat_history.append([message, output])
        return output


if __name__=="__main__":
    inferencer = LLaMa_reader("TheBloke/Llama-2-13B-chat-GPTQ")#TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")

    inferencer.system_prompt="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and as short as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    #"A chat between user and an assistant called llama. The assistant base on history gives helpful, detailed, and polite answers. Use 繁體中文\n. Do not repeat user question."
    def response(s:str):
        ids=[]
        old_string=''
        for i in inferencer.generator(s):
            ids+=[i]
            new_string = inferencer.tokenizer.decode(ids, skip_special_tokens=True)
            print(new_string[-(len(new_string)-len(old_string)):], end='',flush=True)
            old_string=new_string



        return inferencer.generator(s)

    while True:
        s = input("User: ")
        if s != '':
            # inferencer.generate(s)
            print(response(s))
