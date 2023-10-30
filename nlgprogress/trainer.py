from Knowledge_encoder import KnowEncoder
from llama_reader import LLaMa_reader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
E=KnowEncoder()
model=LLaMa_reader("TheBloke/Llama-2-13B-chat-GPTQ")

pooler, emb=E.forward(['hello','world!!'])
emb = emb.to(torch.float16)
print(emb.shape)

print(model.forward(model.tokenizer('hello', return_tensors='pt').input_ids.cuda(), encoder_output=emb.to(device)))
