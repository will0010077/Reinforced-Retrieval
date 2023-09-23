import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, GPT2Tokenizer
from datasets import load_dataset
import json
import numpy as np

class collectgpt():
    def __init__(self, max_len=512, tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len

        self.bos_id=self.tokenizer.bos_token_id
        self.eos_id=self.tokenizer.eos_token_id
        self.pad_id=self.eos_id


    def __call__(self, batch):
        '''
        input:(bs,( (2,(len,len)),1) )
        output: <bos>parent<eos>child<eos>
        '''

        #scores rescale


        tokens=[]
        masks=[]
        targets=[]
        for Q, A in batch:
            q_out=self.tokenizer(Q, return_tensors="pt", truncation=True, max_length=self.max_p_len-2)
            a_out=self.tokenizer(A, return_tensors="pt", truncation=True, max_length=self.max_c_len-1)

            q_ids =q_out['input_ids'][0]
            q_mask=torch.ones_like(q_ids)

            a_ids=a_out['input_ids'][0]
            if a_ids[-1] != self.eos_id:
                a_ids =torch.cat([a_ids,torch.tensor([self.eos_id])])
            a_mask=torch.ones_like(a_ids)

            ids=torch.cat([q_ids,a_ids])

            tokens.append(ids)
            masks.append(torch.cat([q_mask, a_mask]))
            targets.append(torch.cat([torch.ones_like(q_ids)*-100, a_ids]))

        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)
        targets=pad_sequence(targets, batch_first=True, padding_value=-100)
        # print( tokens.shape, masks.shape, targets.shape)#check OK
        return tokens, masks, targets



def read(path):
    dataset= load_dataset('json', data_files=path)
    cols = ['main_tweet','reply','reply_likes']
    dataset=list(zip(*[dataset['train'][c] for c in cols]))
    dataset=np.unique(dataset,axis=0)
    x, y=dataset[:,[0,1]], dataset[:,2]
    y=np.array([int(y) for y in y])
    # y= np.log(y+1)[:,None]
    print('ratio of like>=5:',np.sum(y>=5)/len(y))
    y= (y>=5)[:,None]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01,random_state=43)
    return x_train, x_test, y_train, y_test


if __name__=='__main__':
    read('/home/DS/final/tweets.json')
