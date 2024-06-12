import sys

import torch
from torch import nn
import torch.nn.functional as F
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

from tqdm import tqdm
import yaml
import peft
import os
from socket import socket
token = "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"
# model_dir = "MediaTek-Research/Breeze-7B-Instruct-v0_1"
model_dir = "meta-llama/Llama-2-7b-chat-hf"
bert_dir = "huggingface/bert"
LM_dir = "huggingface/llama2"
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
# torch.autograd.set_detect_anomaly(True)


class collate():
    def __init__(self,):
        
        self.datatokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.LMtokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True, lstrip=False, 
            token='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')
        self.LMtokenizer.pad_token = self.LMtokenizer.eos_token
    def prepare_QA_token(self, texts:list[str], targets:list[str]):
        
        unlabel, label = zip(*[self.templete(q, a) for q,a in zip(texts, targets)])
        cat_qa = [q+" "+a for q, a in zip(unlabel, label)]
        unlabel = self.LMtokenizer(text=unlabel).input_ids
        # print(max([len(s) for s in unlabel]))
        tokens = self.LMtokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=512, truncation =True,)
        
        for i in range(len(texts)):
            tokens['labels'][i, :len(unlabel[i])]=-100
        tokens['labels'][tokens['attention_mask']==0]=-100
        return tokens

    def collate_qa(self, batch:list):
        q_str, a_str = [*zip(*batch)]
        tokens = self.prepare_QA_token(q_str, a_str)
        a_tokens = self.datatokenizer(a_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        return tensor_retuen_type(**tokens), q_str, a_str, tensor_retuen_type(**a_tokens)
    
    def collate_q(self, batch:list):
        batch = [self.templete(q, a) for q, a in batch]
        q_str, a_str = [*zip(*batch)]
        tokens = self.LMtokenizer(text=q_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        a_tokens = self.datatokenizer(a_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        return tensor_retuen_type(**tokens), q_str, a_str, tensor_retuen_type(**a_tokens)
    
    def templete(self, query:str, answer:str ='')->tuple[str]:
        Role = ["system", "user", "assistant"]
        query, answer = query.strip(), answer.strip()
        messages = [
            {"role": "system", "content": "This is the searched knowledge: [KNOW]  [/KNOW] Please answer user questions based on the above knowledge\n"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
        prompt = self.LMtokenizer.apply_chat_template(
            messages[:2],
            tokenize=False, 
            add_generation_prompt=True,
            return_tensors="pt"
        )
        return prompt, answer


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def training(rank, world_size, max_epoch, model, loader, port):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size, port)
    bs = loader.batch_size
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    param_list =[p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(param_list, lr = config['Enc_config']['enc_lr']) #note: Adam work with float16 need to set eps=1e-4 to avoid 0 devided by 0

    iter_step = len(loader)*max_epoch
    warm = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-5, total_iters=int(iter_step*0.02))
    decay =  torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=int(iter_step*1.3), power=1.5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warm, decay], [warm.total_iters])
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts()
    ma_loss=1.8
    stream = torch.cuda.current_stream(rank)
    for epoch in range(max_epoch):
        if rank==10:
            train_bar=tqdm(loader, ncols=0)
        else:
            train_bar = loader
        li=-50
        for i,(tokens, q_str, a_str, a_tokens) in enumerate(train_bar):
            tokens = tokens.to(rank)
            a_tokens = a_tokens.to(rank)
                    

            # feed doc into KnowEnc to get prefix
            if not config['train_config']['use_prefix']:
                a_tokens = None
            
            ref_logp, (LM_output, loss) = model.forward(tokens, Doc_tokens = a_tokens)
            # kl = F.kl_div(LM_output, ref_logp, log_target=True, reduction="batchmean")
            loss = loss.mean()
            # loss += kl.mean() * 0.1


            if config['train_config']['use_prefix']:
                optim.zero_grad()
                loss.backward()
                if i%min(int(128/(world_size*bs)),1)==0:
                    optim.step()
            scheduler.step()

            ma_loss = ma_loss*0.98 + 0.02*(loss if not torch.isnan(loss) else ma_loss)
            if rank==0 and i-li>=50:
                li=i
                print(f'epoch {epoch}, iter {i:3d}, loss: {ma_loss.item():.3f}/{loss.item():.3f}', flush=True)
        stream.synchronize()
        dist.barrier()
        if rank==0:
            torch.save(model.module.state_dict(), "save/EncLM.pt")
            torch.save(optim.state_dict(), "save/EncLM_optim.pt")
        dist.barrier()
        
    cleanup()

def main():
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # init module
    # encoder:torch.Module
    print('Loading LLM')
    LM = LLaMa_reader(LM_dir, 'cpu', token = token, from_pretrained=True)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder=KnowEncoder(dims = num_dims, **config['Enc_config'], dtype=dtype)
    Encoder.to(torch.bfloat16)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=config['Enc_config']['num_layers'], adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    if False:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/EncLM.pt", map_location='cpu'))
    max_epoch = 10
    print('Loading dataset...')
    data_path = "data/cleandata.jsonl"
    dataset = NQADataset(data_path=data_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=collate().collate_qa, persistent_workers=True)
    

    with socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    mp.spawn(training,
        args=(world_size, max_epoch, LM, loader, port),
        nprocs=world_size,
        join=True)
        
        # torch.save(LM.cpu().state_dict(), "save/EncLM.pt") #this is not right!!!
if __name__=="__main__":
    main()
            





