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

def prepare_QA_token(tokenizer, doc:list[str], texts:list[str], targets:list[str]):
    '''
    
    '''
    unlabel= ['\n\n'.join(d_list)+q for d_list, q,a in zip(doc, texts, targets)]
    unlabel = tokenizer(text=unlabel).input_ids
    print(max([len(s) for s in unlabel]))
    cat_qa = ['\n\n'.join(d_list)+q+a for d_list, q,a in zip(doc, texts, targets)]
    tokens = tokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=1024, truncation =True,).to(device)
    
    for i in range(len(texts)):
        tokens['labels'][i, :len(unlabel[i])]=-100
    tokens['labels'][tokens['attention_mask']==0]=-100
    return tokens

if __name__=="__main__":
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
    LM =LLaMa_reader("MediaTek-Research/Breeze-7B-Instruct-v0_1")

    # init agent
    policy = Transformer_Agent(in_dim = 30522)
    policy.to(device)

    optim = torch.optim.AdamW(policy.parameters(), lr = config['train_config']['agent_lr'])
    replay = doc_buffer(max_len=2**10)

    max_epoch = 10
    num_retrieve=1
    num_neg=16
    num_RL_update = 4

    data_path='data/cleandata.pt'
    dataset=NQADataset(data_path=data_path)
    loader = DataLoader(dataset, batch_size = 4, shuffle=True)
    train_bar=tqdm(loader, ncols=0)
    
    ma_loss=30
    ma_reward=-3
    iter=0
    for epoch in range(max_epoch):
        stream = torch.cuda.current_stream()
        for q, target in train_bar:
            iter+=1
            ret = retriever.forward(q)[:,None,:]# (B,1,d)
            outputs = ret
            neg_set=[]
            doc_set = []
            
            with torch.no_grad():
                for k in range(num_retrieve):
                    qt = policy.next(ret) #(B,d)
                    dt, zt = retriever.retrieve(qt, k=num_neg, num_search=4)# [B, neg, 256] [B, neg, 30522]
                    if random.random()<0.05:
                        r = random.randrange(0,num_neg)
                    else:
                        r=0
                    sel_d=dt[:,r,:].unsqueeze(1)
                    sel_z=zt[:,r,:].unsqueeze(1)
                        
                    doc_set.append(sel_d)
                    neg_set.append(zt)
                    ret = torch.cat([ret, sel_z.to(ret.device, non_blocking=True)], dim = 1)
                    outputs = torch.cat([outputs, qt[:,None,:]], dim = 1)
                    
            doc_set = torch.cat(doc_set, dim=1)
            neg_set = torch.stack(neg_set, dim=1) #(B, 5, neg, 256)
            # print(ret.shape, outputs.shape, doc_set.shape)#torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, 256])
            doc = [retriever.tokenizer.batch_decode(doc_set[i]) for i in range(len(doc_set))]

            
            
            
            # !!!forward, loss and reward!!!
            tokens = prepare_QA_token(LM.tokenizer, doc, q, target)
            labels = tokens['labels']
            train_bar.set_postfix_str(f'len: {tokens.input_ids.shape[-1]}')
            # tokens['labels'] = None
            with torch.no_grad():
                y, loss = LM.forward(**tokens)
            # reward=torch.ones([len(q)])
            reward = -loss# temperaly, (B)
            
            
            # update replay buffer
            for i in range(len(q)):
                if torch.isnan(reward[i]):
                    continue
                t = transition(ret[i,:-1], outputs[i,1:], ret[i,1:], neg_set[i], reward[i])
                t.to('cpu', non_blocking=True)
                replay.append(t)
            
            
            if len(replay)>=1024 and iter%20==0:
                stream.synchronize()
                for _ in range(num_RL_update):
                    for t in replay.sample(bs = 64, shuffle=True):
                        optim.zero_grad()
                        t.to(device, non_blocking=True)
                        ma_reward = ma_reward*0.99+t.rewards.mean().item()*0.01
                        t.rewards-=ma_reward
                        # !!!policy update!!!
                        loss = policy.get_loss(t).mean()
                        loss.backward()
                        optim.step()
                        ma_loss = ma_loss*0.99+loss*0.01
                        train_bar.set_postfix_str(f'loss: {ma_loss:.4f}, reward: {ma_reward:.4f}')
                    

            
        





