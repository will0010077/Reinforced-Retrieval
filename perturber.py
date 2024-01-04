import torch
from torch.utils.data import DataLoader

from torch import nn,Tensor
import math
from tqdm import tqdm

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
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class perturb_model(torch.nn.Module):
    
    def __init__(self, in_dim=768, dim=768, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim, dropout=dropout,batch_first=True)
        self.model=torch.nn.TransformerEncoder(self.layer, num_layers)
        # self.model = torch.nn.ModuleList([torch.nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.lam = torch.nn.Parameter(torch.tensor(-10,dtype=torch.float))
        self.pos_encoder = PositionalEncoding(in_dim, dropout=dropout, max_len=128, scale=0.01)
        self.dim=dim
        self.in_dim=in_dim
        if self.dim!=self.in_dim:
            self.scale1=torch.nn.Linear(in_dim, dim, bias=True)
            self.scale2=torch.nn.Linear(dim, in_dim, bias=True)
            self.scale1.weight.data*=1e-2
            self.scale1.weight.data[torch.arange(in_dim),torch.arange(in_dim)]=torch.ones([in_dim])
            self.scale2.weight.data*=1e-2
            self.scale2.weight.data[torch.arange(in_dim),torch.arange(in_dim)]=torch.ones([in_dim])
        
        # for n,p in self.model.named_parameters():
        #     if 'weight' in n and len(p.data.shape)==2:
        #         for i in range(p.data.shape[0]//dim):
        #             p.data[i*dim:(i+1)*dim] = torch.eye(dim, dtype=p.data.dtype)
        #             p.data[i*dim:(i+1)*dim] += torch.randn([dim,dim], dtype=p.data.dtype)*1e-2
                    
    def forward(self, x, mask):
        # x = self.pos_encoder(x)
        if self.dim!=self.in_dim:
            x = self.scale1(x)
        
        x=self.model.forward(x, mask)# + x #  (torch.nn.functional.sigmoid(self.lam))
            
        if self.dim!=self.in_dim:
            x = self.scale2(x)
        return x
    
    
def prepare_parallel(query: Tensor, z: Tensor)->Tensor:
    '''
    query: (B,1,d)
    z: (B,k,d)
    returns: (B,2k,d)
    '''
    query = query.tile(1, z.shape[-2], 1)
    
    return torch.cat([query, z], dim=-2)
    
    
    
    
def parallel_mask(sz = 5, device = 'cpu'):
    '''
    output: shape(2 sz, 2 sz)
    '''
    #get diag=0, other=-inf
    mask1 = torch.full([sz]*2, float('-inf'), dtype=torch.float, device=device)
    mask1[torch.arange(sz), torch.arange(sz)] = 0.0
    
    #get lower tri=0, other=-inf
    mask2 = torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=torch.float, device=device),
        diagonal=0,
    )
    
    mask_top = torch.cat([mask1, mask2], dim=1)#(sz, 2sz)
    
    mask2[torch.arange(sz), torch.arange(sz)] = 0.0
    mask_low = torch.cat([mask1, mask2], dim=1)#(2sz,2sz)
    mask = torch.cat([mask_top, mask_low], dim=0)#(2sz,2sz)
    
    return mask
    
     
    
    

if __name__=='__main__':
    device='cuda'
    dim=32
    model=perturb_model(in_dim=dim, dim = 1024, num_heads=8, num_layers=4, dropout=0.0)
    model.to(device)
    
    num_sample=2**13
    reduce=2**10
    L=5
    Bs=256
    
    
    mask = parallel_mask(L, device)
    #toy sample
    #random generate sample with shape [num_sample, Length, dim]
    train_x = torch.randn([num_sample, 1, dim],device=device)
    train_y = train_x + torch.randn([num_sample, L, dim], device=device)*0.1 
    print('shape of x: ',train_x.shape)
    
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5, betas=[0.8,0.99], weight_decay=0.01)
    lr_schedu = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1, total_iters=100)
    train_loader=DataLoader(train_x, batch_size=Bs, shuffle=True)
    
    losses = 1
    num_epoch = [0, 40]
    bar = tqdm(range(num_epoch[0]), ncols=100)
    for epoch in bar:
        bar.set_description_str(f'epoch: {epoch}/{num_epoch[0]}')
        lr_schedu.step()
        for i, x in enumerate(train_loader):
            lr_schedu.step()
            y = x + torch.randn([x.shape[0], L, dim], device=device)*0.1 
            x_ = prepare_parallel(x, y)
            out = model.forward(x_, mask)
            #MSE with target shift
            loss:Tensor = (out[:,:L,:] - x).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses = 0.98*losses + 0.02*loss.item()
            bar.set_postfix_str(f'loss: {losses: .4f}, lam: {model.lam.item(): 7.1e}')
            
    print(f'step 1: {losses: .6f}')
    
    
    train_y = train_x + torch.randn([num_sample//reduce, L, dim], device=device).tile(reduce,1,1)*0.1 
    train_loader=DataLoader(torch.cat([train_x,train_y],dim=1), batch_size=Bs, shuffle=True)
    
    bar = tqdm(range(num_epoch[1]), ncols=100)
    for epoch in bar:
        bar.set_description_str(f'epoch: {epoch}/{num_epoch[1]}')
        lr_schedu.step()
        for i, x in enumerate(train_loader):
            lr_schedu.step()
            y = x[:,1:]
            x = x[:,:1]
            x_ = prepare_parallel(x, y)
            out = model.forward(x_, mask)
            #MSE with target shift
            loss:Tensor = (out[:,:L,:] - y).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses = 0.98*losses + 0.02*loss.item()
            bar.set_postfix_str(f'loss: {losses: .4f}, lam: {model.lam.item(): 7.1e}')
    print(f'step 2: {losses: .6f}')
    
        
