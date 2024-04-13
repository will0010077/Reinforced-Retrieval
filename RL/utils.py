# import numpy as np
import sys
sys.path.append('..')
import torch
from torch import Tensor, nn
import math
from DocBuilder.Retriever_k_means import inner
import random
class transition(nn.Module):
    def __init__(self, inputs:Tensor, preds:Tensor, ret:Tensor, neg:Tensor, rewards:Tensor):
        '''
        inputs : (k,d)
        pred : (k,d)
        ret : (k,d)
        neg : (k,neg,d)
        rewards : (k) or scalar
        '''
        super().__init__()
        self.register_buffer('inputs', inputs.detach().clone())
        self.register_buffer('preds', preds.detach().clone())
        self.register_buffer('ret', ret.detach().clone())
        self.register_buffer('neg', neg.detach().clone())
        self.register_buffer('rewards', rewards.detach().clone())
        self.inputs:Tensor
        self.preds:Tensor
        self.ret:Tensor
        self.neg:Tensor
        self.rewards:Tensor
        
        
    def __str__(self) -> str:
        return f'inputs:{self.inputs.shape}, outputs:{self.preds.shape}, ret:{self.ret.shape}, neg:{self.neg.shape}, rewards:{self.rewards.shape}'


class doc_buffer:
    def __init__(self, max_len=2**14):
        self.clear()
        self.max_len=max_len
    
    def append(self, t):
        '''
        adding a transition to buffer
        '''
        self.buffer.append(t)
        if len(self)>self.max_len:
            self.buffer.pop(0)
    
    def stack(self, name, s:Tensor = None):
        if s is not None:
            return torch.stack([getattr(self.buffer[i], name) for i in s])
        return torch.stack([getattr(x, name) for x in self.buffer])
    
    def sample(self, bs, shuffle = False):
        if shuffle:
            index = torch.randperm(len(self))
        else:
            index = torch.arange(len(self))
        
        for i in range(0, len(self), bs):
            yield transition(self.stack('inputs', index[i:i+bs]), 
                          self.stack('preds', index[i:i+bs]), 
                          self.stack('ret', index[i:i+bs]),
                          self.stack('neg', index[i:i+bs]),
                          self.stack('rewards', index[i:i+bs]))
        
    def __len__(self,):
        return len(self.buffer)
    def clear(self,):
        self.buffer = []


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000, scale=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe*scale
        self.register_buffer('pe', pe)
        self.pe:Tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = torch.cat([x, self.pe[:,:x.size(1)].tile(x.shape[0],1,1)], dim = -1)
        return self.dropout(x)
    
class perturb_model(nn.Module):
    
    def __init__(self, in_dim=768, dim=768, num_heads=4, num_layers=2, dropout=0.1, pos_dim = 64):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim, dropout=dropout,batch_first=True)
        self.model=torch.nn.TransformerEncoder(self.layer, num_layers)
        # self.model = torch.nn.ModuleList([torch.nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.lam = torch.nn.Parameter(torch.tensor(-10,dtype=torch.float))
        self.pos_encoder = PositionalEncoding(pos_dim, dropout=dropout, max_len=16, scale=0.01)
        self.dim=dim
        self.in_dim=in_dim
        
        self.scale1=torch.nn.Linear(in_dim + pos_dim, dim, bias=True)
        self.scale2=torch.nn.Linear(dim, in_dim, bias=True)
                    
    def forward(self, x:torch.Tensor, mask=None)->Tensor:
        '''
        x: (B,n,d)
        mask: (n,n)
        out: shape of x
        '''
        x = self.pos_encoder(x)
        x = self.scale1(x)
        
        if mask is None:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device)
        x=self.model.forward(x, mask)# + x #  (torch.nn.functional.sigmoid(self.lam))
            
        x = self.scale2(x)
        return x
    
    
class Transformer_Agent(nn.Module):
    def __init__(self,in_dim, dim=768):
        super().__init__()
        self.model = perturb_model(in_dim, dim)
        self.lam = nn.Parameter(torch.ones(1)*-10, True)
    def forward(self, x):
        y=self.model.forward(x)
        return self.lam.exp()*torch.log(1+y.relu_())+x
    
    @torch.no_grad()
    def next(self, x:torch.Tensor):
        '''
        x: (B,n,d)
        output: (B,d)
        '''
        x = self.forward(x)
        
        return x[:,-1,:]
    
    def get_loss(self, t:transition)->Tensor:
        '''
        Reinforce algorithm
        return : loss (B,k)
        '''
        t.neg#(([32, 5, 16, 30522]))
        outputs = self.forward(t.inputs)#(32,5,30522)
        neg = (outputs[:,:,None,:] * t.neg).sum(-1)*10
        M = torch.max(neg, dim=-1, keepdim=True).values#(32,5,1)
        log_pi = (outputs * t.ret).sum(-1)*10 - M[:,:,0] - (neg-M).exp().sum(-1).log()
        
        loss = -t.rewards[:,None] * log_pi + 0.1*torch.abs(outputs).sum(-1)
        # regularization to original input query
        loss += 0.1*((outputs - t.inputs)**2).sum(-1)
        return loss    

if __name__=='__main__':
    
    B = doc_buffer()
    
    for i in range(10000):
        B.append(transition(torch.rand([5,64]), torch.rand([5,64]), torch.rand([5,64]),torch.rand([5,64]), torch.ones(1)))
    B.clear()
    for i in range(10000):
        B.append(transition(torch.rand([5,64]), torch.rand([5,64]), torch.rand([5,64]),torch.rand([5,64]), torch.ones(1)))
    
    print(B.sample())
    