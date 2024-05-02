import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask
from LM.llama_reader import LLaMa_reader
from LM.Knowledge_encoder import KnowEncoder
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

def templete(doc_list:list[str], query:str, answer:str)->tuple[str]:
    doc_list = '\n\n'.join(doc_list)
    prompt = f'''<s> <<SYS>>\n This is the searched knowledge: [KNOW] {doc_list} [\KNOW]
    Please answer user questions based on the above knowledge\n<</SYS>>
    \n [INST] User:{query.strip()} [/INST] Assistant:'''
    return prompt, prompt + answer
def prepare_QA_token(tokenizer, doc:list[list[str]], texts:list[str], targets:list[str]):
    '''
    
    '''
    
    unlabel, cat_qa = zip(*[templete(doc_list, q, a) for doc_list, q,a in zip(doc, texts, targets)])
    unlabel = tokenizer(text=unlabel).input_ids
    # print(max([len(s) for s in unlabel]))
    tokens = tokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=1024, truncation =True,).to(device)
    
    for i in range(len(texts)):
        tokens['labels'][i, :len(unlabel[i])]=-100
    tokens['labels'][tokens['attention_mask']==0]=-100
    return tokens

if __name__=="__main__":
    device='cuda'
    

    # pre compute embbeding


    # init module
    # encoder:torch.Module
    LM =LLaMa_reader("MediaTek-Research/Breeze-7B-Instruct-v0_1")
    num_heads = LM.model.config.num_key_value_heads
    num_layers = LM.model.config.num_hidden_layers
    num_dims = LM.model.config.hidden_size//LM.model.config.num_attention_heads
    print(LM.model.config)
    Encoder=KnowEncoder(num_layers, num_heads, num_dims, config['train_config']['head'])
    Encoder.to(device)
    
    prefix, prefix_masks = Encoder.forward(['hello'], k = 1, dtype=torch.float16)
    tokens = LM.tokenizer(['world'], return_tensors='pt').to(device)
    LM.forward(**tokens, encoder_output=prefix, encoder_masks=prefix_masks)
    
    # init agent
    policy = Transformer_Agent(in_dim = 30522, num_heads=6, num_layers=4, pos_dim=128)
    policy.to(device)

    optim = torch.optim.AdamW(policy.parameters(), lr = config['train_config']['agent_lr'])
    replay = doc_buffer(max_len=2**10)

# init retriever
    cluster_config=config["cluster_config"]
    cluster = cluster_builder(k=cluster_config["k"])
    cluster.load('05_02_19_08')

    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.to('cpu')
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever895.pt', map_location='cpu')['enc_model_state_dict'])

    data=torch.load('data/data_reduced_1000000.pt') ## shape:(N,d)
    retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
    retriever.to(device)
    
    
    max_epoch = 10
    num_retrieve=4
    num_neg=32
    num_RL_update = 8

    data_path='data/cleandata.pt'
    dataset=NQADataset(data_path=data_path)
    dataset.data=dataset.data[:4]*10000
    loader = DataLoader(dataset, batch_size = 4, shuffle=True)
    
    ma_loss=10
    ma_reward=-2
    iter=0
    
    for epoch in range(max_epoch):
        train_bar=tqdm(loader, ncols=0)
        stream = torch.cuda.current_stream()
        for q, target in train_bar:
            B = len(q)
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
                    
            doc_set = torch.cat(doc_set, dim=1)#(B, 5, 256)
            doc_set = doc_set.reshape([-1, doc_set.shape[-1]])#(B*k, 256)
            
            neg_set = torch.stack(neg_set, dim=1) #(B, 5, neg, 256)
            # print(ret.shape, outputs.shape, doc_set.shape)#torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, 256])
            # doc = [retriever.tokenizer.batch_decode(doc_set[i]) for i in range(len(doc_set))]

            mask = generate_mask(doc_set, Encoder.tokenizer.pad_token_id)
            doc_set=tensor_retuen_type(input_ids = doc_set, attention_mask = mask).to(device)
            prefix, prefix_masks = Encoder.forward(doc_set, k = num_retrieve, dtype=torch.float16)
            # !!!forward, loss and reward!!!
            tokens = prepare_QA_token(LM.tokenizer, [[]]*B, q, target)
            labels = tokens['labels']
            train_bar.set_postfix_str(f'len: {tokens.input_ids.shape[-1]}')
            # tokens['labels'] = None
            y, loss = LM.forward(**tokens, encoder_output=prefix, encoder_masks=prefix_masks)
            del y
            prefix.retain_grad()
            
            loss.backward()
            print(prefix.grad)
            # reward=torch.ones([len(q)])
            reward = -loss# temperaly, (B)
            # update replay buffer
            for i in range(len(q)):
                if torch.isnan(reward[i]):
                    continue
                t = transition(inputs=ret[i,:-1], outputs = outputs[i,1:],ret = ret[i,1:], neg = neg_set[i], rewards = reward[i]).to_sparse()
                t.to('cpu', non_blocking=True)
                replay.append(t)
            
            
            if len(replay)>=128 and iter%20==0:
                stream.synchronize()
                for _ in range(num_RL_update):
                    for t in replay.sample(bs = 64, shuffle=True):
                        optim.zero_grad()
                        t.to(device)
                        t.to_dense()
                        ma_reward = ma_reward*0.98+t.rewards.mean().item()*0.02
                        # !!!policy update!!!
                        pi_loss, v_loss, reg_loss, flops_loss = policy.get_loss(t)
                        loss = (pi_loss + v_loss+ 0.005*reg_loss).mean() # + 0.0001*flops_loss
                        loss.backward()
                        optim.step()
                        ma_loss = ma_loss*0.99+loss*0.01
                        train_bar.set_postfix_str(f'loss: {pi_loss.mean():.4f}/{v_loss.mean():.4f}/{reg_loss.mean():.4f}/{ma_loss:.4f}, reward: {ma_reward:.4f}')
                    

            
        





