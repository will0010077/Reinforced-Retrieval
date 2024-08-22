import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
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
from DatasetLoader.dataset import PretrainEnc
from DatasetLoader.collate_func import collate
from config import bert_dir, LM_dir, token, enc_config, train_config

from tqdm import tqdm
import yaml
import peft
import os
from socket import socket

# torch.autograd.set_detect_anomaly(True)




def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def training(rank, world_size, start_epoch, max_epoch, model, dataset, collate_fn, port):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size, port)
    start, end = int(len(dataset)*rank/world_size), int(len(dataset)*(rank+1)/world_size)
    loader = DataLoader(dataset[start: end], batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn, persistent_workers=True)

    bs = loader.batch_size
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    Enc_param_list =[p for n, p in model.named_parameters() if p.requires_grad and "adaption_" not in n]
    Prefix_param_list =[p for n, p in model.named_parameters() if p.requires_grad and "adaption_" in n]
    optim = torch.optim.AdamW([{"params":Enc_param_list, "lr":enc_config.enc_lr}, {"params":Prefix_param_list, "lr":enc_config.prefix_lr}], betas = train_config.betas, weight_decay = 0.05) #note: Adam work with float16 need to set eps=1e-4 to avoid 0 devided by 0
    # optim.load_state_dict(torch.load("save/EncLM_optim.pt", map_location="cpu"))
    iter_step = len(loader)*max_epoch
    warm = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-5, total_iters=int(iter_step*0.05))
    decay =  torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=int(iter_step*1.2), power=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warm, decay], [warm.total_iters])
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts()
    ma_loss=2
    stream = torch.cuda.current_stream(rank)
    optim.zero_grad()
    for epoch in range(start_epoch, start_epoch+max_epoch):
        if rank==0:
            train_bar=tqdm(loader, ncols=0)
        else:
            train_bar = loader
        li=-50
        for i,(qa_tokens, d_tokens) in enumerate(train_bar):
            qa_tokens = qa_tokens.to(rank)
            d_tokens = d_tokens.to(rank)
                    

            # feed doc into KnowEnc to get prefix
            if not train_config.use_prefix:
                d_tokens = None
            
            ref_logp, (LM_output, loss) = model.forward(qa_tokens, Doc_tokens = d_tokens, stage = int(epoch>=1))
            # kl = F.kl_div(LM_output, ref_logp, log_target=True, reduction="batchmean")
            loss = loss.mean()
            (loss).backward()
            # loss += kl.mean() * 0.1
            if i%max(int(32/(world_size*bs)),1)==0:
                # Unscale the gradients and perform optimizer step
                optim.step()

                # Update the scaler
            scheduler.step()
# stop rolling ! 
            ma_loss = ma_loss*0.95 + 0.05*(loss.item() if not torch.isnan(loss) else ma_loss)
            if rank==0 and i-li>=1:
                li=i
                train_bar.set_postfix_str(f'epoch {epoch}, iter {i:3d}/{len(train_bar)}, loss: {ma_loss:.3f}/{loss.item():.3f}')
                # print(f'epoch {epoch}, iter {i:3d}/{len(train_bar)}, loss: {ma_loss.item():.3f}/{loss.item():.3f}', flush=True)
        stream.synchronize()
        dist.barrier()
        if rank==0:
            torch.save(model.module.state_dict(), f"save/NQ_EncLM_{epoch}.pt")
            torch.save(optim.state_dict(), "save/EncLM_optim.pt")
        dist.barrier()
        
    cleanup()

def main():
    '''Train the knowledge encoder by (x,y,d) pairs, the data is from data_preprocess.py with "train_QAD" postfix, most be done before RL training.'''
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
    Encoder=KnowEncoder(dims = num_dims, **enc_config, dtype=dtype)
    Encoder.train()
    # Encoder.to(torch.bfloat16) # !!!!use autocast instead of model.to(bfloat16)!!!!

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=32, adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    if True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        try:
            LM.load_state_dict(torch.load("save/TV_EncLM_0.pt", map_location='cpu'), strict=True)
        except:
            print("Loading may not finish.")
        
    start_epoch = 0
    max_epoch = 3 # update epoch, not end epoch
    print('Loading dataset...')
    
    if False:
        data_path = "data/TV_train_QAD.jsonl"
        dataset = PretrainEnc(data_path=data_path, use_doc=True, use_short=True, use_long=False, num_samples = None)
        # dataset = [*dataset]*2**7
        length = 128
        collate_fn = collate(LM_dir, bert_dir, max_length=length, form="short").collate_qa_docs
    else:
        data_path = "data/NQ_train_QAD.jsonl"
        dataset = PretrainEnc(data_path=data_path, use_doc=True, use_short=False, use_long=True, num_samples = None)
        length = 256
        collate_fn = collate(LM_dir, bert_dir, max_length=length, form = "long").collate_qa_docs
    dataset = [*dataset]

    with socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    mp.spawn(training,
        args=(world_size, start_epoch, max_epoch, LM, dataset, collate_fn, port),
        nprocs=world_size,
        join=True)
        
        # torch.save(LM.cpu().state_dict(), "save/EncLM.pt") #this is not right!!!
if __name__=="__main__":
    main()
            





