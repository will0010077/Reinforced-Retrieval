# import numpy as np
import torch
from torch import Tensor, nn
import math

class transition:
    def __init__(self, inputs:Tensor, preds:Tensor, ret:Tensor, rewards:Tensor):
        '''
        inputs : (k,d)
        pred : (k,d)
        ret : (k,d)
        rewards : (k) or scalar
        '''
        self.inputs, self.preds, self.ret, self.rewards = inputs.cpu(), preds.cpu(), ret.cpu(), rewards.cpu()

    def __str__(self) -> str:
        return f'inputs:{self.inputs.shape}, outputs:{self.preds.shape}, ret:{self.ret.shape}, rewards:{self.rewards.shape}'


class doc_buffer:
    def __init__(self,):
        self.clear()
    
    def append(self, t):
        '''
        adding a transition to buffer
        '''
        self.buffer.append(t)
        pass
    
    def stack(self, i, s:slice = None):
        
        return torch.stack([getattr(x, i) for x in self.buffer])
    
    def sample(self,):
        
        return transition(self.stack('inputs'), 
                          self.stack('preds'), 
                          self.stack('ret'), 
                          self.stack('rewards'))
        
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
    
class perturb_model(torch.nn.Module):
    
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
        
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    @torch.no_grad()
    def next(self, x:torch.Tensor, mask=None):
        '''
        x: (B,n,d)
        output: (B,d)
        '''
        x = self.model.forward(x, mask)
        
        return x[:,-1,:]
        

if __name__=='__main__':
    
    B = doc_buffer()
    
    for i in range(10000):
        B.append(transition(torch.rand([5,64]), torch.rand([5,64]), torch.rand([5,64]), torch.ones(1)))
    B.clear()
    for i in range(10000):
        B.append(transition(torch.rand([5,64]), torch.rand([5,64]), torch.rand([5,64]), torch.ones(1)))
    
    print(B.sample())
    