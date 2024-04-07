import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.LexMAE import lex_retriever
from LM.llama_reader import LLaMa_reader
from fintune_contriver import NQADataset
import yaml

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

'''

Given document D, dataset (q_i,y_i)
Compute latents Z={z_1,..., z_N\} = Emb(d_1)
While not converge
----Given a query $q$, hidden state z_q = Emb(q)
----for t in 1 to k:
--------Kt=Vt=Concat(z_q,z1,...,z(t-1))
--------Perturbed embedding zt Transformer(z_q, Kt, Vt)
--------top n subset D_q=top n(D,z_q^t)
--------Retrieve document (dt,zt) from policy.
----end for
----get y hat = LM(q, d1,...,dk)
----loss = CE(y hat, y)
----update policy and value loss
end while

'''
device='cuda'
# init retriever

cluster_config=config["cluster_config"]
cluster = cluster_builder(k=cluster_config["k"])
cluster.load('04_06_15_54')

lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to('cpu')
lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever838.pt', map_location='cpu')['enc_model_state_dict'])

data=torch.load('data/data_reduced_10000.pt') ## shape:(N,d)
retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
retriever.to(device)

# pre compute embbeding


# init module
# encoder:torch.Module
# LM =LLaMa_reader("MediaTek-Research/Breeze-7B-Instruct-v0_1")

# init agent
policy = Transformer_Agent(in_dim = 30522)
policy.to(device)

optim = torch.optim.AdamW(policy.parameters(), lr = config['train_config']['agent_lr'])
replay = doc_buffer()

max_epoch = 10
num_retrieve=5
num_RL_update = 200

data_path='data/cleandata.pt'
dataset=NQADataset(data_path=data_path)
loader = DataLoader(dataset, batch_size = 8, shuffle=True)
train_bar=tqdm(loader, ncols=0)
for epoch in range(max_epoch):
    for q, target in train_bar:
        ret = retriever.forward(q)[:,None,:]# (B,1,d)
        outputs = ret
        d_set=[]
        z_set=[]
        doc_set = []
        
        for k in range(num_retrieve):
            
            qt = policy.forward(ret)[:,-1] #(B,d)
            dt, zt = retriever.retrieve(qt, k=1)# [B, topk, 256] [B, topk, 30522]
            doc_set.append(dt)
            ret = torch.cat([ret, zt.to(ret.device)], dim = 1)
            outputs = torch.cat([outputs, qt[:,None,:]], dim = 1)
            
        doc_set = torch.cat(doc_set, dim=1)
        # print(ret.shape, outputs.shape, doc_set.shape)::torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, 256])
        
        
        # !!!forward, loss and reward!!!
        with torch.no_grad():
            y, loss = LM.forward(q, target)
        reward=torch.ones([len(q)])
        # reward = -loss# temperaly, (B)
        
        
        # update replay buffer
        for i in range(len(q)):
            replay.append(transition(ret[i,:-1], outputs[i,1:], ret[i,1:], reward[i]))
        
        if len(replay)>500:
            print(replay.sample())
            for _ in range(num_RL_update):
                # !!!policy update!!!
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            replay.clear()
        
    





