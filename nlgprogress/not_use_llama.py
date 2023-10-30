
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sep='</s>'
eos='</s>'
class TaiwanLLaMaGPTQ:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        new_line=[[self.tokenizer('\n\n').input_ids[-1]]]
        unicode=[[i] for i in range(3,259)]
        self.generate_config=dict(
                            max_new_tokens = 1024,
                            # top_k = 50,
                            # top_p = 0.9,
                            # temperature  =  0.7,
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
            texts.append('response.strip()+')
        texts.append('[INST]  '+message.strip()+'[/INST]')
        texts.append(' ')
        return (sep).join(texts)

    def generate(self, message: str):
        prompt = self.get_prompt(message, self.chat_history)
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids
        generate_ids = self.model.generate(input_ids=tokens.cuda(), streamer=self.streamer, **self.generate_config)
        output = self.tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        self.chat_history.append([message, output])
        return output

    def thread_generate(self, message:str):
        from threading import Thread
        prompt = self.get_prompt(message, self.chat_history)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=1024,
            streamer=self.thread_streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in self.thread_streamer:
            yield new_text

        thread.join()


inferencer = TaiwanLLaMaGPTQ("TheBloke/Llama-2-13B-chat-GPTQ")#TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")

inferencer.system_prompt="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and as short as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

#"A chat between user and an assistant called llama. The assistant base on history gives helpful, detailed, and polite answers. Use 繁體中文\n. Do not repeat user question."
def response(s:str):

    return inferencer.generate(s)

if __name__=="__main__":
    while True:
        s = input("User: ")
        if s != '':
            print(response(s))
