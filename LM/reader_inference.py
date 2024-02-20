
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sep='</s>'
eos='</s>'
class LLaMa_reader:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False)
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token_id
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
        # print(self.model)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.thread_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def get_prompt(self, message: str, chat_history: list[tuple[str, str]]) -> str:
        texts = [f'{self.system_prompt}']
        if self.external is not None:
            texts.append(self.external)
        for user_input, response in chat_history[-1:]:
            texts.append('[INST] '+user_input.strip()+'[/INST]')
            texts.append(response.strip())
        texts.append('[INST] '+message.strip()+'[/INST]')
        texts.append(' ')
        return (sep).join(texts)

    def generator(self, message, encoder_output = None, max_new_tokens = 1024,):
        prompt = self.get_prompt(message, self.chat_history)
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        for n in range(max_new_tokens):
            if n>0:
                tokens = torch.tensor([[generate_ids]],device=self.model.device)
                encoder_output = output.past_key_values
            output=self.model.forward(input_ids = tokens ,past_key_values=encoder_output, use_cache=True )
            generate_ids = torch.argmax(output.logits[0,-1], dim=0)
            for i in range(len(output.past_key_values)):
                for j in range(len(output.past_key_values[i])):
                    print(f'({i}, {j})',output.past_key_values[i][j].shape)#(1,40,2,128)->(B,head,n,dim)

            text = self.tokenizer.decode(generate_ids, lstrip=False)

            yield text
            if generate_ids == self.eos_id:
                break
    def generate(self, message: str):
        prompt = self.get_prompt(message, self.chat_history)
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids

        generate_ids = self.model.generate(input_ids=tokens.cuda(), past_key_values=1, streamer=self.streamer, **self.generate_config)

        output = self.tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        self.chat_history.append([message, output])
        return output


inferencer = TaiwanLLaMaGPTQ("TheBloke/Llama-2-13B-chat-GPTQ")#TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")

inferencer.system_prompt="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and as short as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

#"A chat between user and an assistant called llama. The assistant base on history gives helpful, detailed, and polite answers. Use 繁體中文\n. Do not repeat user question."
def response(s:str):
    for i in inferencer.generator(s):
        print(i, end=' ')
    return inferencer.generator(s)

if __name__=="__main__":
    while True:
        s = input("User: ")
        if s != '':
            # inferencer.generate(s)
            print(inferencer.generate(s))
