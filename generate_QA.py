from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
def QAtemplate(documents):
    return [f'''Here is a document:\n\n<\d>{doc}<\d>\n\nPlease ask a question about this document and provide answer in json format: {{"Q": [your question here], "A": [your answer here]}}<eos>''' for doc in documents]

system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and as short as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.'''
system_prompt = '''You are a machine that only provide json format {"Q": [your question here], "A": [your answer here]}, carefully read the document, only answer what is mentioned in the document, you can directly copy the content in the document'''
def main():
    #loading document for generate QA
    doc_set:list
    with open(f'data/smart_factory.jsonl','r',encoding='unicode_escape') as f:
        doc_set=json.load(f)


    doc_set = preprocess(doc_set)

    model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_dir =  "meta-llama/Llama-2-7b-chat-hf"

    hug_face_key = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=hug_face_key, use_fast=True, lstrip=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    LM = AutoModelForCausalLM.from_pretrained(model_dir, token=hug_face_key, device_map='cuda', use_cache=True, torch_dtype=torch.bfloat16)
    streamer=None
    # tokenizer.model_max_length=2048
    # eos_id = tokenizer.eos_token_id
    # eos = tokenizer.eos_token
    # tokenizer.pad_token_id = eos_id
    loader = DataLoader(doc_set, batch_size=8)
    bar = tqdm(loader)
    for documents in bar:
        print(QAtemplate(documents)[0])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QAtemplate(documents)[0]},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(LM.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        print(tokenizer.decode(input_ids[0]))
        outputs = LM.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(output)
        exit()

        


if __name__=='__main__':
    main()