import sys
sys.path.append("..")
# Load model directly
from DocBuilder.Retriever_k_means import cluster_builder
from DocBuilder.utils import top_k_sparse, generate_mask, sparse_retrieve_rep, max_pooling, cos_sim
import torch
from torch import Tensor

from transformers import AutoTokenizer, AutoModel,AutoModelWithLMHead, BertTokenizerFast
from transformers import BertConfig, BertLMHeadModel,AutoModelForMaskedLM
import logging
from functools import reduce
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import random
import yaml
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
model_config = config['model_config']

seed = config['seed']


class lex_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    def forward(self, x:dict, output_low=False)->list[torch.Tensor]:
        '''
        x: dict[input_ids, attention_mask]
        output: Tensor[B, d]
        '''
        mask = x.get('attention_masks',None)
        if mask is None:
            mask=generate_mask(x['input_ids'], self.tokenizer.pad_token_id)
        output=self.model(input_ids = x.get('input_ids',None), attention_mask = mask, output_hidden_states=True)
        logits, hidden_state = output.logits, output.hidden_states[-1]
        if output_low:
            return hidden_state
        
        a = max_pooling(logits, mask)
        a = torch.softmax(a, dim=1) #(B,30522)

        Word_embedding=self.model.bert.embeddings.word_embeddings.weight ## Grad-disallowed
        b = (a @ Word_embedding.detach()) #(B,30522)* (30522, 768,)=(B,768)

        return logits, hidden_state, b


class lex_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Initializing a BERT google-bert/bert-base-uncased style configuration
        configuration = BertConfig()
        configuration.num_hidden_layers=2
        configuration.is_decoder=False
        # print(configuration)
        # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
        self.tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        self.model = BertLMHeadModel(configuration)


    def forward(self, x:dict, b:torch.Tensor=None)->torch.Tensor:
        '''
        x: dict[input_ids, attention_mask]
        output: Tensor[B, d]
        '''
        mask = x.get('attention_mask',None)
        if mask is None:
            mask=generate_mask(x['input_ids'], self.tokenizer.pad_token_id)
        x = self.model.bert.embeddings(x.get('input_ids',None))
        if b is not None:
            x[:,0] = b # input bottleneck
        y = self.model(inputs_embeds = x, attention_mask = mask)
        y = y.logits

        return y

class lex_retriever(torch.nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.model=lex_encoder()
        self.tokenizer=self.model.tokenizer
        
        if out_dim is not None:
            self.proj = torch.nn.Linear(768, out_dim)

    def forward(self, x:dict, output_soft:bool=False):
        mask = x.get('attention_mask',None)
        if mask is None:
            mask=generate_mask(x['input_ids'], self.tokenizer.pad_token_id)
            
        logits, hidden_state, b= self.model.forward(x, output_low=False)

        if output_soft:
            return torch.softmax(max_pooling(logits, mask), dim=-1)
        return sparse_retrieve_rep(max_pooling(logits, mask))
        return sparse_retrieve_rep(b)
        return sparse_retrieve_rep(max_pooling(hidden_state, mask))
        return b

    @torch.inference_mode()
    def get_feature(self, ids, bs)->torch.Tensor:
        '''
        return: tensor with shape:(N, 768)
        '''
        self.train()
        temp = self.collate([ids[0]])
        temp['input_ids'] = temp['input_ids'].to(self.model.model.device)
        feature_shape  = self.forward(temp).shape[1]
        feature_list=[]


        dataloader = DataLoader(ids, batch_size=bs, shuffle=False, collate_fn=self.collate, num_workers=8)
        for i,idx in (bar:=tqdm(enumerate(dataloader),ncols=0,total=len(dataloader))):
            idx['input_ids'] = idx['input_ids'].to(self.model.model.device)
            feature  = self.forward(idx)#(bs, d)

            # sparse_feature = [top_k_sparse(v, 128).cpu() for v in feature]
            sparse_feature = top_k_sparse(feature, config['cluster_config']['k_sparse']).cpu() # (bs, d) with sparse
            feature_list.append(sparse_feature) # (len, bs, d)
        return  torch.cat(feature_list)
    def collate(self, ids):
        ids = torch.stack(ids)
        return {'input_ids':ids}

if __name__=='__main__':
    enc=lex_encoder()
    dec=lex_decoder()
    x=enc.tokenizer(['where is taiwan?','DOTDOTDOT'],return_tensors='pt', padding=True)
    # print(x)
    enc_logits, hidden_embed, b=enc.forward(x)
    print(enc_logits.shape, hidden_embed.shape, b.shape)
    dec_logits=dec.forward(x, b)
    print(dec_logits.shape)
