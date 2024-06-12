from Knowledge_encoder import KnowEncoder
from llama_reader import LLaMa_reader
from contriver import DOC_Retriever

import torch
import torch.utils.checkpoint
import random
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda'

import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)

@torch.inference_mode()
def demo(query):
    E.eval()
    topkseg=retriever.retrieve(query, k=train_config['topk'])
    topkseg=topkseg.reshape(-1,topkseg.shape[-1])
    topkseg=topkseg.to(device)
    x={'input_ids':topkseg,'attention_mask': torch.ones_like(topkseg).to(device)}
    
    _, embs, embsmasks=E(x=x, k=train_config['topk'], dtype=torch.float16)
    
    predict=model.generate('The answer of the question: '+query+' is', embs, embsmasks, 16, test=False,streamer=False)
    predict=predict.strip()
    
    return predict
        
def response(s:str):
        return demo(s)

if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['loader_config']
   
    retriever = DOC_Retriever()
    model=LLaMa_reader("meta-llama/Llama-2-7b-chat-hf")

    num_heads = model.model.config.num_key_value_heads
    num_layers = model.model.config.num_hidden_layers
    num_dims = model.model.config.hidden_size//num_heads

    E=KnowEncoder(num_layers, num_heads, num_dims, train_config['head'])
    E.to(device)
    E.model.requires_grad_(False)
    model.tokenizer.padding_side='left'

    E.load_state_dict(torch.load('/home/devil/workspace/nlg_progress/backend/app/save/knowled_encoder.pt'))

    
    while True:
        s = input("User: ")
        if s != '':
            # inferencer.generate(s)
            print(response(s))

    