import torch
from tqdm import tqdm
from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.LexMAE import lex_retriever
from LM.llama_reader import LLaMa_reader
from fintune_contriver import NQADataset
import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
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
# init retriever

cluster_config=config["cluster_config"]
cluster = cluster_builder(k=cluster_config["k"])
cluster.load('03_31_20_22')

lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to('cpu')
lex_MAE_retriver.model.load_state_dict(torch.load('app/save/LEX_MAE_retriever838.pt', map_location='cpu')['enc_model_state_dict'])

data=torch.load('app/data/data_reduced_200000.pt') ## shape:(N,d)
retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
# retriever.to('cuda')

# pre compute embbeding


# init module
# encoder:torch.Module
LM = LLaMa_reader('')

# init agent
policy = Transformer_Agent(dim = 768)

optim = torch.optim.AdamW(policy.parameters(), lr = config['train_config']['agent_lr'])
replay = doc_buffer()

max_epoch = 10
num_retrieve=5

data_path='app/data/cleandata.pt'
dataset=NQADataset(data_path=data_path)
train_bar=tqdm()
for epoch in range(max_epoch):
    for q, target in train_bar:
        ret = retriever.forward(q)
        outputs = ret
        d_set=[]
        z_set=[]
        for k in num_retrieve:
            
            qt = policy.forward(ret)[:,-1] #(B,d)
            dt, zt = retriever.retrieve(qt)
            doc_set.append(dt)
            ret = torch.cat([ret, zt[:,None,:]], dim = 1)
            outputs = torch.cat([outputs, qt[:,None,:]], dim = 1)
            
        doc_set = list(zip(doc_set))
        #forward and loss
        with torch.no_grad():
            y, loss = LM.forward(q, target)
        
        reward = -loss# temperaly, (B)
        
        # grad update
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        
        # replay buffer
        for i in range(len(q)):
            replay.append(transition(ret[i], outputs[i,1:], ret[i,:-1], reward[i]))
        
        if len(replay)%5000 ==0:
            policy.update()
            replay.clear()
        
    





