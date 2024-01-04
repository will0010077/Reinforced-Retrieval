
import numpy as np
import torch
import torch.nn as nn
from model import gptmodel, Retriever, select_Bk
from dataprepare import collectgpt, read
from tqdm import tqdm
from transformers import GPT2Tokenizer
import time

use_retrieve=True
use_rl=True
device='cuda:0' if torch.cuda.is_available() else 'cpu'
def get_p_from_logit(out, target, mask):
    out=torch.softmax(out, dim=2)
    i1=np.arange(out.shape[0])[:,None]
    i2=np.arange(out.shape[1])[None,:]
    pred=out[i1,i2,target]#(B,L)
    pred[target==-100]=0
    pred[mask==0]=0
    return pred

def train(model:gptmodel, retriever:Retriever, bar:tqdm, optim:torch.optim.AdamW, RLoptim:torch.optim.AdamW, warmup:torch.optim.lr_scheduler.LambdaLR):
    '''
    bar: tqdm(DataLoader)
    '''
    
    model.train()
    losses=3
    rewards=0.
    reward_is=0
    acc=[]
    k=4
    mini=64*k
    for QA,query in bar:
        '''
        tokens:(bs, len)
        masks:(bs, len)
        target:(bs, len)
        '''

        optim.zero_grad()
        RLoptim.zero_grad()
        bs=len(query)
        start=time.time()
        if use_retrieve:
            index=retriever.retrieve(query, k)
            retrieved = select_Bk(retriever.texts, index)#(B,k)
            hiddens=retriever.hiddens(retrieved, index)

            with torch.no_grad():
                rand_index=torch.randint(0, retriever.knowledge.shape[0], [bs, 8])
                retrieved = select_Bk(retriever.texts, rand_index)#(B,k)
                old=retriever.knowledge[rand_index]
                retriever.hiddens(retrieved, rand_index)
                new=retriever.knowledge[rand_index]
                diff=((old-new)**2).mean()

        else:
            hiddens=None
        retri_time=time.time()-start

        # for qa, r in zip(QA, retrieved):
        #     qa[0] = ''.join(r)+qa[0]

        tokens, masks, target=c_func(QA)
        tokens=tokens.to(device)
        masks=masks.to(device)
        target=target.to(device)

        start=time.time()
        out, loss = model.forward(tokens, masks, hiddens, target) #(bs, len, vocab), (bs, 1)
        out=out.detach()[:,-tokens.shape[1]:-1,:]
        with torch.no_grad():
            out_wo, _ = model.forward(tokens, masks,) #(bs, len, vocab), (bs, 1)
        out_wo=out_wo[:,:-1,:]
        masks=masks[:,1:]
        target=target[:,1:]

        RL_loss=0
        #reward function
        if use_retrieve and use_rl:
            pred=get_p_from_logit(out,target,masks)
            pred_wo=get_p_from_logit(out_wo,target,masks)
            reward=pred.sum(dim=1)/pred_wo.sum(dim=1)-1
            reward_i=pred_wo.sum(dim=1)/pred.sum(dim=1)-1

            # rand_index=torch.randint(0, retriever.knowledge.shape[0], [mini])
            # rand_knowledge=retriever.knowledge[rand_index][None,...]#(1,n, 768)
            # rand_knowledge=rand_knowledge.expand([bs,-1,-1])#(bs,n,768)
            #hidden:(bs,k,768),
            # logpi, value=retriever.pi_v(torch.cat([hiddens, rand_knowledge], dim=1), query)#(bs,k+n)
            logpi, value=retriever.pi_v(hiddens, query)#(bs,k+n)
            # logpi, value=logpi[:,:k], value[:,:k]
            adv=reward[:,None]-value
            RL_loss = - torch.mean((adv.detach()-0.02)*logpi) + torch.mean(adv**2)


        #update
        (loss+RL_loss).backward()
        if warmup.last_epoch<1000:
            warmup.step()
        optim.step()
        RLoptim.step()
        lm_time=time.time()-start

        #log
        losses = losses*(1-0.02)+loss.item()*0.02
        rewards=rewards*(1-0.02)+(reward.mean())*0.02
        reward_is=reward_is*(1-0.02)+(reward_i.mean())*0.02
        log_str=f'loss: {losses:5.4f}'#, time:{int(retri_time*1000):3d}, {int(lm_time*1000):3d} ms
        if use_rl:
            log_str+=f', dif:{diff:5.3f}, r: {rewards:5.3f}, {reward_is:5.3f}'
        bar.set_postfix_str(log_str)#v: {adv.detach().mean():5.3f}

    return np.mean(losses)

def templete(x):
    return 'Tweet: '+x[0]+'Reply: ', x[1]+' '

if __name__=='__main__':
    x_train, x_test, y_train, y_test=read('tweet_reply.json')
    x_train=x_train[:2000]
    x_train=list(map(templete, x_train))
    pure_text=list(map(lambda x: x[0]+x[1], x_train))
    model=gptmodel().to(device)
    retriever=Retriever(use_sim= not use_rl).to(device)
    if use_retrieve:
        retriever.update(pure_text)
        torch.save(retriever.knowledge,'embedding.pt')
        retriever.texts=pure_text
        retriever.knowledge=torch.load('embedding.pt')
    # index=retriever.retrieve(['hello, welcome to New York!'], k=10)

    # model.load_state_dict(torch.load('gptmodel_001.pt', map_location='cpu'))

    c_func=collectgpt(max_len=256, tokenizer = GPT2Tokenizer.from_pretrained('gpt2'))
    train_cfg={
        'batch_size':    8,
        'shuffle':       True,
        'num_workers':   4,
        'collate_fn':    lambda x: [x,[q for q,a in x]],
    }

    train_loader=torch.utils.data.DataLoader(x_train, **train_cfg)

    optim=torch.optim.AdamW(list(model.parameters())+list(retriever.AC.encoder.parameters())+list(retriever.emb.parameters()) , lr= 0e-5, betas=[0.9,0.98],weight_decay=0.01)
    RLoptim=torch.optim.AdamW(list(retriever.AC.actor.parameters())+list(retriever.AC.critic.parameters()), lr= 0e-5, betas=[0.9,0.98],weight_decay=0.01)

    warmup=torch.optim.lr_scheduler.LambdaLR(optim, lambda i: i/100 if i<100 else 1)
    start=0
    num_epoch=30
    for epoch in range(start+1,start+num_epoch+1):
        bar=tqdm(train_loader, ncols=100)
        bar.set_description_str(f'epoch:{epoch:03d}')
        loss=train(model,retriever, bar, optim,RLoptim, warmup)

        if epoch%1==0:
            torch.save(model.state_dict(), f'gptmodel_{epoch:03d}.pt')


