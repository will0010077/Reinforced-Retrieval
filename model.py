
from transformers import RobertaModel, RobertaTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import vmap
from tqdm import tqdm

class TextEncoder(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net=RobertaModel.from_pretrained("roberta-base")
        self.tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
        self.bs=16

    def forward(self, texts):
        '''
        text: (B, len)
        output:(B, d)
        '''
        out_list=[]
        bar=range(0, len(texts), self.bs)
        for i in bar:
            batch=texts[i:i+self.bs]

            token=self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
            ids = token['input_ids'].to(self.net.device)#(B,len)
            masks=token['attention_mask'].to(self.net.device)
            if ids[0,0]!=self.tokenizer.cls_token_id:
                cls_vec=torch.ones([ids.shape[0],1], device=ids.device, dtype=torch.long)*self.tokenizer.cls_token_id
                ids=torch.cat([cls_vec, ids], dim=1)
                cls_mask_vec=torch.ones([ids.shape[0],1], device=ids.device, dtype=torch.long)
                masks=torch.cat([cls_mask_vec,masks],dim=1)

            x=self.net(input_ids=ids, attention_mask=masks)
            out=x['last_hidden_state'][...,0,:].clone()
            del x
            out_list.append(out)
        return torch.cat(out_list)


class RetrieveAC(nn.Module):
    def __init__(self, in_dim=768, hidden=256, use_sim=False) -> None:
        super().__init__()
        self.use_sim=use_sim
        self.in_dim=in_dim
        self.hidden=hidden
        self.encoder=TextEncoder()
        self.encoder.eval()
        self.actor=nn.ParameterDict({'emb':torch.randn([in_dim,hidden],requires_grad=True),
                                     'doc':torch.randn([in_dim,hidden],requires_grad=True)})

        self.critic=nn.Sequential(nn.Linear(self.in_dim, hidden),
                               nn.LeakyReLU(inplace=True),
                               nn.Linear(hidden,1))

    def scoring(self, knowledge, h_q):
        '''
        knowledge:(N,d) or (B,N,d)
        h_q:(B,d)
        out:(B,N)
        '''
        if len(knowledge.shape)==2:
            knowledge=knowledge[None,:,:]


        if self.use_sim:
            ele=knowledge*h_q[:,None,:]#(B,N,d)
            out=ele.sum(dim=2)#similarity
        else:
            ele=(knowledge@self.actor.doc)*(h_q@self.actor.emb)[:,None,:]#(B,N,d)
            out=ele.sum(dim=2)#similarity
        return out

    def value(self, knowledge, h_q):
        '''
        knowledge:(N,d)
        h_q:(B,d)
        out_logpi:(B,N)
        '''
        if len(knowledge.shape)==2:
            knowledge=knowledge[None,:,:]
        ele=knowledge*h_q[:,None,:]#(B,N,d)
        out=self.critic(ele)[:,:,0]#(B,N)
        return out
    def logpi(self, knowledge, h_q):
        '''
        knowledge:(N,d) or (B,N,d)
        h_q:(B,d)
        out_logpi:(B,N)
        '''
        score=self.scoring(knowledge, h_q)
        # logpi=F.log_softmax(score, dim=1)
        return score


class Retriever(nn.Module):
    def __init__(self, use_sim=False):
        super().__init__()
        self.knowledge=None
        self.texts=None
        self.AC=RetrieveAC(use_sim=use_sim)
        self.emb=nn.Linear(768,768, bias=False)

    @torch.no_grad()
    def update(self, texts, index=None, embedding=None):
        '''
        texts:(N,:)
        encoder:nn.module
        self.knowledge:(N,d)
        '''
        if index is not None:
            assert embedding is not None
            self.knowledge[index]=embedding.clone()
        else:
            self.knowledge=self.AC.encoder(texts)
            self.texts=texts

    @torch.no_grad()
    def retrieve(self, query, k:int):
        '''query:B x text, k: num of retrieve, output (index(B,k))'''
        h_q=self.AC.encoder(query)#(B,d)
        logpi=self.AC.logpi(self.knowledge, h_q)#(B,N)
        values, indices = torch.topk(logpi, k=k, dim=1)

        return indices

    def hiddens(self, keys, indexs):
        '''key, indexs= (B,N)[text].\
        out: (B,N)'''
        bs=len(keys)
        hiddens=[]
        keys=sum(keys,[])
        indexs=indexs.reshape([-1])
        # for key, index in zip(keys, indexs):
        hiddens=self.AC.encoder(keys)
        hiddens=self.emb(hiddens)
        self.update(keys, indexs, hiddens.detach())
        hiddens=hiddens.reshape([bs,-1,hiddens.shape[-1]])
        return hiddens


    def pi_v(self, hiddens, query):
        '''hiddens:(B,N,d)\
        query: (B)[text]'''
        # hiddens=hiddens.detach()
        # with torch.no_grad():
        q=self.AC.encoder(query)#(B,d)
        logpi=self.AC.logpi(hiddens, q)#(B,N)
        value=self.AC.value(hiddens, q)
        return logpi, value






class gptmodel(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net  = GPT2LMHeadModel.from_pretrained('gpt2')
    def forward(self, ids, masks, embeds=None, target=None):

        inputs_embeds = self.net.transformer.wte(ids)
        if embeds is not None:
            padding=torch.ones_like(embeds[:,:,0], device=embeds.device, dtype=torch.long)
            masks = torch.cat([padding, masks], dim=1)
            target = torch.cat([-100*padding, target], dim=1)
            inputs_embeds= torch.cat([embeds, inputs_embeds], dim=1)
        output = self.net(inputs_embeds=inputs_embeds, attention_mask=masks, labels=target)

        return output.logits, output.loss

class gptactor(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net  = GPT2LMHeadModel.from_pretrained('gpt2')
        self.critic= nn.Sequential(nn.Linear(self.net.get_output_embeddings().in_features, 128),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(128, 1))
    def forward(self, ids, masks, target=None):
        '''
        input: id, masks:(bs, len)

        output: prob:(bs, len, vocab)
        '''
        out=[]
        output = self.net(input_ids=ids, attention_mask=masks, output_hidden_states =True, labels=target)
        value=self.critic(output.hidden_states[-1])
        out.append(output.logits)
        out.append(value)
        if target is not None:
            out.append(out.loss)
        return out

def select_Bk(x, index):
    '''x:(N), index:(B,k), out:(B,k)'''
    return [[x[i] for i in j]for j in index]
if __name__=='__main__':
    t=RobertaTokenizer.from_pretrained("roberta-base")
    ids=t(['hello','world!'], padding=True)
    print(ids)
