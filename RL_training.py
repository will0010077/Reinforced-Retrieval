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
    prompt = f'''<<SYS>>\n This is the searched knowledge: [KNOW] {doc_list} [\KNOW]
    Please answer user questions based on the above knowledge\n<</SYS>>
    \n [INST] User: {query.strip()} [/INST] Assistant: '''
    return prompt, prompt + answer
def prepare_QA_token(tokenizer, doc:list[list[str]], texts:list[str], targets:list[str]):
    '''
    
    '''
    
    unlabel, cat_qa = zip(*[templete(doc_list, q, a) for doc_list, q,a in zip(doc, texts, targets)])
    unlabel = tokenizer(text=unlabel).input_ids
    # print(max([len(s) for s in unlabel]))
    tokens = tokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=256, truncation =True,).to(device)
    
    for i in range(len(texts)):
        tokens['labels'][i, :len(unlabel[i])]=-100
    tokens['labels'][tokens['attention_mask']==0]=-100
    return tokens

if __name__=="__main__":
    device='cuda:0'
    

    print('Initilize Agent')
    policy = Transformer_Agent(in_dim = 30522, dim = 512, num_heads=8, num_layers=2)
    policy.to(device)
    RL_optim = torch.optim.AdamW(policy.parameters(), lr = config['train_config']['agent_lr'])
    replay = doc_buffer(max_len=2**10)
    
    cluster_config=config["cluster_config"]
    cluster = cluster_builder(k=cluster_config["k"])
    cluster.load('05_02_19_08')
    # pre compute embbeding

    # init module
    # encoder:torch.Module
    print('Loading LLM')
    LM =LLaMa_reader("MediaTek-Research/Breeze-7B-Instruct-v0_1", device)
    num_heads = LM.model.config.num_key_value_heads
    num_layers = LM.model.config.num_hidden_layers
    num_dims = LM.model.config.hidden_size//LM.model.config.num_attention_heads
    # print(LM.model.config)
    
    
    print('Initialize KnowEnc...')
    Encoder=KnowEncoder(num_layers, num_heads, num_dims, config['train_config']['head'], num_prefix=4)
    pretrain_Encoder:dict = torch.load('save/LEX_MAE_retriever_loss_6.8032.pt', map_location='cpu')['dec_model_state_dict']
    for k in [*pretrain_Encoder.keys()]:
        if 'predictions' not in k:
            pretrain_Encoder['.'.join(k.split('.')[2:])] = pretrain_Encoder[k]
        del pretrain_Encoder[k]
    pretrain_Encoder['pooler.dense.weight'] = Encoder.model.pooler.dense.weight
    pretrain_Encoder['pooler.dense.bias'] = Encoder.model.pooler.dense.bias
    Encoder.model.load_state_dict(pretrain_Encoder, assign=True)
    Encoder.to(device)
    del pretrain_Encoder
    Enc_optim = torch.optim.AdamW(Encoder.parameters(), lr = config['train_config']['enc_lr'])
    
    print('testing prefix tuning...')
    prefix, prefix_masks = Encoder.forward(['hello'], k = 1, dtype=torch.float16)
    tokens = LM.tokenizer(['world'], return_tensors='pt').to(device)
    LM.forward(**tokens, encoder_output=prefix, encoder_masks=prefix_masks)
    


    # init retriever

    print('Initilize retriever')
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=True)
    data=torch.load('data/data_reduced_10000000.pt') ## shape:(N,d)
    retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
    retriever.to(device)
    retriever.model.to(device)
    retriever.model.device=device
    del lex_MAE_retriver, data, cluster
    
    
    max_epoch = 10
    num_retrieve=4
    num_neg=16
    num_RL_update = 8

    
    print('Loading dataset...')
    data_path='data/cleandata.pt'
    dataset=NQADataset(data_path=data_path)
    # dataset.data=dataset.data[:4]*10000
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
            
            
            # send to gpu to retrieval loop
            # policy.to(device)
            ret = retriever.forward(q)[:,None,:]# (B,1,d)
            outputs = ret
            neg_set=[]
            doc_set = []
            
            with torch.no_grad():
                #retrieval loop
                for k in range(num_retrieve):
                    qt = policy.next(ret) #(B,d)
                    dt, zt = retriever.retrieve(qt, k=num_neg, num_search=4)# [B, neg, n] [B, neg, 30522]
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
            # policy.to('cpu')
            # retriever.to('cpu')
            
            # pos neg pair reshape
            doc_set = torch.cat(doc_set, dim=1)#(B, 5, n)
            doc_set = doc_set.reshape([-1, doc_set.shape[-1]])#(B*k, n)
            
            neg_set = torch.stack(neg_set, dim=1) #(B, 5, neg, n)
            
            # print(ret.shape, outputs.shape, doc_set.shape)#torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, n])
            
            # doc = [retriever.tokenizer.batch_decode(doc_set[i]) for i in range(len(doc_set))]

            # feed doc into KnowEnc to get prefix
            if config['train_config']['use_prefix']:
                mask = generate_mask(doc_set, Encoder.tokenizer.pad_token_id)
                doc_set = tensor_retuen_type(input_ids = doc_set, attention_mask = mask).to(device)
                prefix, prefix_masks = Encoder.forward(doc_set, k = num_retrieve, dtype=torch.float16)
            else:
                prefix, prefix_masks = None, None
            # QA pre-process
            tokens = prepare_QA_token(LM.tokenizer, [[]]*B, q, target)
            labels = tokens['labels']
            
            # print(LM.tokenizer.batch_decode(tokens.input_ids))
            # labels[labels==-100]=0
            # print(LM.tokenizer.batch_decode(labels))
            
            
            # !!!LLM prefix tuning forward, loss and reward!!!
            y, loss = LM.forward(**tokens, encoder_output=prefix, encoder_masks=prefix_masks)
            del y
            
            reward = -loss.detach()# temperaly, (B)
            loss = loss.mean()
            
            if config['train_config']['use_prefix']:
                Enc_optim.zero_grad()
                loss.backward()
                Enc_optim.step()
            train_bar.set_postfix_str(f'len: {tokens.input_ids.shape[-1]}, loss: {loss.item():.3f}, reward: {ma_reward:.2f}')
            
            if loss.item()>50:
                continue
            # update replay buffer
            for i in range(len(q)):
                if torch.isnan(reward[i]):
                    continue
                t = transition(inputs=ret[i,:-1], preds = outputs[i,1:], ret = ret[i,1:], neg = neg_set[i], rewards = reward[i]).to_sparse()
                t = t.to('cpu', non_blocking=True)
                replay.append(t)
            
            # RL update
            if len(replay)>=128 and iter%20==0:
                stream.synchronize()
                for _ in range(num_RL_update):
                    for t in replay.sample(bs = 32, shuffle=True):
                        RL_optim.zero_grad()
                        t = t.to(device, non_blocking=True)
                        t = t.to_dense()
                        ma_reward = ma_reward*0.98+t.rewards.mean().item()*0.02
                        # !!!policy update!!!
                        pi_loss, v_loss, reg_loss, flops_loss = policy.get_loss(t)
                        del flops_loss
                        loss = (pi_loss + v_loss+ 1*reg_loss).mean() 
                        loss.backward()
                        RL_optim.step()
                        ma_loss = ma_loss*0.99+loss*0.01
                        train_bar.set_postfix_str(f'loss: {pi_loss.mean():.3f}/{v_loss.mean():.3f}/{reg_loss.mean():7.2e}/{ma_loss:.3f}, reward: {ma_reward:.2f}')
                    del t
                    

            
        





