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
    device = 1
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
    collate_qa = collate().collate_qa
    while True:
        x = input("Question: ")
        y = input("Answer: ")
        tokens, unlabel, unlabel_str,  q_str, a_str, a_tokens= collate_qa([(x,y)])
        tokens = tokens.to(device)
        a_tokens = a_tokens.to(device)
        
        with torch.no_grad():
            p_generation = LM.pseudo_generate(tokens, a_tokens)
            print("Pseudo generation: ", p_generation)
                

            message = unlabel_str #+" "+" ".join(a_str[j].split()[:5])
            output_pre = LM.generate(message, Doc_tokens=a_tokens, max_new_tokens=256)
            output_ori = LM.generate(message, Doc_tokens=None, max_new_tokens=256)
            
            print("prefix: ", output_pre,'\n', "ori: ", output_ori)

        ori_bert=Bert_score(output_ori, list(a_str))[0]
        pre_bert=Bert_score(output_pre, list(a_str))[0]
        pseudo_bert=Bert_score(p_generation, list(a_str))[0]

        ori_bleu=BLEU_score(output_ori, list(a_str))[0]
        pre_bleu=BLEU_score(output_pre, list(a_str))[0]
        pseudo_bleu=BLEU_score(p_generation, list(a_str))[0]
        
        print(f"BERT: ori: {ori_bert:.3f}/prefix: {pre_bert:.3f}/pseudo: {pseudo_bert:.3f}\nBLEU: ori: {ori_bleu:.3f}/prefix: {pre_bleu:.3f}/pseudo: {pseudo_bleu:.3f}")