import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file, load_file


from DocBuilder.utils import tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder
from DatasetLoader.dataset import NQADataset
from metric.reward import BLEU_score, Bert_score

from tqdm import tqdm
import yaml
import peft
import os

from PrefixPretrain import collate
token = "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"
model_dir = "meta-llama/Llama-2-7b-chat-hf"

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
if __name__=="__main__":
    device = 0
    print('Loading LLM')
    LM = LLaMa_reader(model_dir, 'cpu', token = token, from_pretrained=True)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder=KnowEncoder(dims = num_dims, **config['Enc_config'], dtype=dtype)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=config['Enc_config']['num_layers'], adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    LM.to(device)
    LM.eval()

    if True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/EncLM.pt", map_location='cpu'))
    max_epoch = 1
    print('Loading dataset...')
    data_path = "data/cleandata.jsonl"
    dataset = NQADataset(data_path=data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate().collate_q, persistent_workers=True)

    for i,(tokens, q_str, a_str, a_tokens) in enumerate(loader):
        
        tokens = tokens.to(device)
        a_tokens = a_tokens.to(device)
        
        with torch.no_grad():
            prefix = LM.Enc.forward(a_tokens)
            message = [q_str[j]+" "+" ".join(a_str[j].split()[:5]) for j in range(len(q_str))]
            LM_output = LM.generate(message, prefix=prefix, max_new_tokens=256)

        LM_output = [ LM_output[j][len(q_str[j]):] for j in range(len(q_str))]
        
        print(LM_output, list(a_str))
        print("bert:", Bert_score(LM_output, list(a_str)))
        if i==10:
            break