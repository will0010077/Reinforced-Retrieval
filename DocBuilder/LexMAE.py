import sys
sys.path.append("../../")
sys.path.append("app/lib/DocBuilder/")
# Load model directly
from DocBuilder.Retriever_k_means import cluster_builder, cos_sim
import torch
from torch import Tensor

from transformers import AutoTokenizer, AutoModel,AutoModelWithLMHead
from transformers import BertConfig, BertLMHeadModel,AutoModelForMaskedLM
import logging
from functools import reduce
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import random
import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
model_config = config['model_config']

seed = config['seed']


def special_token_mask_generation(input_ids, special_token_ids):
    init_no_mask = torch.full_like(input_ids, False, dtype=torch.bool)
    mask_bl = reduce(lambda acc, el: acc | (input_ids == el),
                     special_token_ids, init_no_mask)
    return mask_bl.to(torch.long)

def text_part_mask_generation(input_ids, special_token_ids, attention_mask):
    mask_text_part = (1 - special_token_mask_generation(input_ids, special_token_ids)) * attention_mask
    return mask_text_part

def exp_mask(_mask, _val, high_rank=False):
    _exp_mask = (torch.ones_like(_mask) - _mask).to(_val.dtype) * \
                torch.full([1], fill_value=-10000, dtype=_val.dtype, device=_val.device)
    if high_rank:
        _exp_mask = _exp_mask.unsqueeze(-1).expand_as(_val)
    return _exp_mask + _val


def zero_mask(_mask, _val, high_rank=False):
    _zero_mask = _mask.to(_val.dtype)
    if high_rank:
        _zero_mask = _zero_mask.unsqueeze(-1).expand_as(_val)
    return _zero_mask * _val

def masked_pool(rep_input, rep_mask, high_rank=True, method="mean", return_new_mask=False):

    dim_pool = rep_mask.dim() - 1
    new_mask = (rep_mask.sum(dim=dim_pool) > 0).to(rep_mask.dtype)

    if method == "mean":
        masked_input = zero_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = masked_input.sum(dim=dim_pool)
        denominator = rep_mask.to(rep_output.dtype).sum(dim=dim_pool)
        # remove zero
        denominator = torch.where(
            denominator > 0.,
            denominator, torch.full_like(denominator, fill_value=1.)
        )
        if high_rank:
            denominator = denominator.unsqueeze(-1).expand_as(rep_output)
        rep_output /= denominator

    elif method == "max":
        masked_input = exp_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = torch.max(masked_input, dim=dim_pool)[0]
    else:
        raise NotImplementedError

    rep_output = zero_mask(new_mask, rep_output, high_rank=high_rank)

    if return_new_mask:
        return rep_output, new_mask
    else:
        return rep_output

def generate_bottleneck_repre(
        input_ids, attention_mask, bottleneck_src,
        special_token_ids=None, word_embeddings_matrix=None,
        last_hidden_states=None, mlm_logits=None,
):
    if bottleneck_src == "cls":
        bottleneck_repre = last_hidden_states[:, 0].contiguous()
    elif bottleneck_src.startswith("logits"):
        with torch.no_grad():
            mask_text_part = text_part_mask_generation(input_ids, special_token_ids, attention_mask)
        pooled_enc_logits = masked_pool(mlm_logits, mask_text_part, high_rank=True, method="max")  # bs,V
        # mlm_logits.masked_fill_((mask_text_part == 0).unsqueeze(-1), 0.)  # apply mask
        # pooled_enc_logits = torch.max(mlm_logits, dim=1).values
        if bottleneck_src == "logits":
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1)  # bs,V
        elif bottleneck_src == "logits_sat":
            pooled_enc_saturated_logits = torch.log(torch.relu(pooled_enc_logits) + 1.)  # bs,V
            pooled_enc_probs = pooled_enc_saturated_logits / (
                    pooled_enc_saturated_logits.sum(-1, keepdim=True) + 1e-4)
        else:
            raise NotImplementedError(bottleneck_src)
        bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix.detach())    # bs,h
    else:
        raise NotImplementedError(bottleneck_src)
    return bottleneck_repre

batch_sparse = torch.func.vmap(torch.Tensor.to_sparse)
def top_k_sparse(x:torch.Tensor, k:int, vec_dim:int=-1):
    '''
    x: Tensor
    vec_dim: data dim, default -1
    out: sparsed x
    '''
    scale=len(x.shape)*2+1
    if scale>(x.shape[vec_dim]/k):
        print(f'Warning! Sparsed result larger than original Tensor. scale: {scale}, sparsity: {(x.shape[vec_dim]/k)}')
    assert k<=x.shape[vec_dim]# check k smaller than original size
    a, _=x.argsort(dim=vec_dim).split_with_sizes(split_sizes=[x.shape[vec_dim]-k, k], dim=vec_dim) #keep top k index
    x=x.scatter(dim=vec_dim, index=a, value=0)#other index full with zero
    x=(x).to_sparse()
    return x

def generate_mask(x, pad:int):
    '''
    x:(B,N) with pad
    output: mask extend one token
    '''
    mask = (x['input_ids']!=pad).long()
    mask:Tensor
    front = torch.ones([len(mask),1], dtype=torch.long, device=mask.device)
    mask = torch.cat([front, mask], dim=-1)[:,:-1]
    return mask

class lex_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    def forward(self, x:dict, output_low=False)->list[torch.Tensor]:
        '''
        x: dict[input_ids, attention_mask]
        output: Tensor[B, d]
        '''
        mask = x.get('attention_mask',None)
        if mask is None:
            mask=generate_mask(x, self.tokenizer.pad_token_id)
        output=self.model(input_ids = x.get('input_ids',None), attention_mask = mask, output_hidden_states=True)
        logits, hidden_state = output.logits, output.hidden_states[-1]
        if output_low:
            return hidden_state
        
        a = self.max_pooling(logits, mask)
        a = torch.softmax(a, dim=1) #(B,30522)

        Word_embedding=self.model.bert.embeddings.word_embeddings.weight ## Grad-disallowed
        b = (a @ Word_embedding.detach()) #(B,30522)* (30522, 768,)=(B,768)

        return logits, hidden_state, b

    def Lex_mae_pooling(self, y, mask):
        return torch.log(1+torch.relu(self.max_pooling(y, mask)))

    def max_pooling(self, token_embeddings:Tensor, mask:Tensor):
        token_embeddings.masked_fill_(~mask.bool()[..., None], float('-inf'))
        sentence_embeddings = torch.max(token_embeddings, dim=1)
        return sentence_embeddings.values


class lex_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Initializing a BERT google-bert/bert-base-uncased style configuration
        configuration = BertConfig()
        configuration.num_hidden_layers=2
        configuration.is_decoder=True
        # print(configuration)
        # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = BertLMHeadModel(configuration)


    def forward(self, x:dict, b:torch.Tensor=None)->torch.Tensor:
        '''
        x: dict[input_ids, attention_mask]
        output: Tensor[B, d]
        '''
        mask = x.get('attention_mask',None)
        if mask is None:
            mask=generate_mask(x, self.tokenizer.pad_token_id)
        x = self.model.bert.embeddings(x.get('input_ids',None))
        if b is not None:
            x[:,0] = b # input bottleneck
        y = self.model(inputs_embeds = x, attention_mask = mask)
        y = y.logits

        return y

class lex_retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=lex_encoder()
        self.tokenizer=self.model.tokenizer

    def forward(self, x:dict):
        mask = x.get('attention_mask',None)
        if mask is None:
            mask=generate_mask(x, self.tokenizer.pad_token_id)
            
        logits, hidden_state, b= self.model.forward(x, output_low=False)


        return self.model.Lex_mae_pooling(logits, mask)
        return self.model.max_pooling(logits, mask)
        return self.model.Lex_mae_pooling(hidden_state, mask)
        return b
        return torch.log(1+torch.relu(b))

    @torch.inference_mode()
    def get_feature(self, ids, bs)->torch.Tensor:
        '''
        return: tensor with shape:(N, 768)
        '''
        feature_shape  = self.forward(self.collate([ids[0]])).shape[1]
        feature_ts=torch.empty([len(ids), feature_shape],dtype=torch.float32)


        dataloader = DataLoader(ids, batch_size=bs, shuffle=False, collate_fn=self.collate,num_workers=0)
        for i,idx in (bar:=tqdm(enumerate(dataloader),ncols=0,total=len(dataloader))):
            feature  = self.forward(idx)#(bs, d)


            feature_ts[i*bs: i*bs+bs ,:]=(feature)
        return  feature_ts
    def collate(self, ids):
        ids = torch.stack(ids)
        return {'input_ids':ids.to(self.model.model.device)}

if __name__=='__main__':
    enc=lex_encoder()
    dec=lex_decoder()
    x=enc.tokenizer(['where is taiwan?','DOTDOTDOT'],return_tensors='pt', padding=True)
    # print(x)
    enc_logits, hidden_embed, b=enc.forward(x)
    print(enc_logits.shape, hidden_embed.shape, b.shape)
    dec_logits=dec.forward(x, b)
    print(dec_logits.shape)
