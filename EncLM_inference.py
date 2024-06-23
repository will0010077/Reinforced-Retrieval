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
    dataset = NQADataset(data_path=data_path, num_samples=10000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate().collate_qa, persistent_workers=True)

    ori_bert_list = []
    pre_bert_list = []
    ori_bleu_list = []
    pre_bleu_list = []
    f = open("moniter.txt", "a")
    for i,(tokens, unlabel, unlabel_str,  q_str, a_str, a_tokens) in enumerate(loader):
        
        tokens = tokens.to(device)
        del tokens['labels']
        tokens.input_ids = tokens.input_ids[:,:128]
        tokens.attention_mask = tokens.attention_mask[:,:128]
        a_tokens = a_tokens.to(device)
        
        with torch.no_grad():
            prefix = LM.Enc.forward(a_tokens)
            message = [unlabel_str[j] for j in range(len(unlabel_str))] #+" "+" ".join(a_str[j].split()[:5])
            output_pre = LM.generate(message, prefix=prefix, max_new_tokens=256)
            output_ori = LM.generate(message, prefix=None, max_new_tokens=256)

        # output_ori = [ output_ori[j][len(q_str[j]):] for j in range(len(q_str))]
        # output_pre = [ output_pre[j][len(q_str[j]):] for j in range(len(q_str))]

        ori_bert=Bert_score(output_ori, list(a_str))
        pre_bert=Bert_score(output_pre, list(a_str))

        ori_bleu=BLEU_score(output_ori, list(a_str))
        pre_bleu=BLEU_score(output_pre, list(a_str))

        ori_bert_list+=ori_bert
        pre_bert_list+=pre_bert

        ori_bleu_list+=ori_bleu
        pre_bleu_list+=pre_bleu

        for j in range(len(q_str)):
            f.write(f"Prompt: {message[j]}\nGround truth: {a_str[j]}\n[{ori_bert[j]:.3f}, {ori_bleu[j]:.3f}]Original response: {output_ori[j]}\n[{pre_bert[j]:.3f}, {pre_bleu[j]:.3f}]Prifix response: {output_pre[j]}\n"+"="*80+"\n")
        
        # print(LM_output, list(a_str))
        if i==10:
            break
    print("ori bert:", sum(ori_bert_list)/len(ori_bert_list))
    print("prefix bert:", sum(pre_bert_list)/len(pre_bert_list))
    print("ori bleu:", sum(ori_bleu_list)/len(ori_bleu_list))
    print("prefix bleu:", sum(pre_bleu_list)/len(pre_bleu_list))