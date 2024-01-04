import torch
from torch import nn, Tensor
import math
from tqdm import tqdm
import torch.utils.data.dataloader

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, scale=0.1):
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
    
class latent_transformer(torch.nn.Module):
    
    def __init__(self, in_dim, dim=0, forward_dim = 0, num_heads=4, num_layers=2, dropout = 0.1):
        """
        in_dim: input dim
        dim: embed dim, default = in_dim
        forward_dim: dim of feedforward layers, default = in_dim
        num_heads: number of attention heads(it is better if dim per heads(dim/num_heads) >=64)
        num_layers: number of layers
        """
        super().__init__()
        if dim==0:
            dim = in_dim
        if forward_dim==0:
            forward_dim = in_dim
            
        self.in_dim=in_dim
        self.dim=dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if self.dim!=self.in_dim:
            self.scale1=torch.nn.Linear(in_dim, dim, bias=True)
        self.scale2=torch.nn.Linear(dim, in_dim, bias=True)
        self.layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=num_heads, dim_feedforward=forward_dim, dropout=dropout, batch_first=True)
        self.model=torch.nn.TransformerEncoder(self.layer, self.num_layers)
        self.pos_encoder = PositionalEncoding(dim, dropout = dropout, max_len=4096, scale=0.01)
        
            
        
    def forward(self, x:Tensor):
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device)
        
        if self.dim!=self.in_dim:
            x = self.scale1(x)

        # add position encoding
        x_pos = self.pos_encoder.forward(x)
        
        # feed into transformer
        out = self.model.forward(x_pos, mask) 
            
        out = self.scale2(out)
            
        return out
    
    @torch.no_grad()
    def generate(self, x:Tensor, max_new_len=1024):
        '''
        x: shape(B,n,dim)
        
        '''
        for i in range(max_new_len):
            out = self.forward(x)
            x = torch.concat([x, out[:,-1:,:]], dim=-2)
        
        return x
            
    


if __name__=='__main__':
    device='cpu'
    device='cuda'
    
    dim=64
    model=latent_transformer(in_dim=dim, dim = 512, forward_dim =512, num_heads=512//128, num_layers=4, dropout=0)
    model.to(device)
    
    num_sample=1024
    L=5
    Bs=32
    
    
    #toy sample
    #random generate sample with shape [num_sample, Length, dim]
    train_x = torch.randn([num_sample, 1, dim],device=device).tile(1,L,1)
    #add some noise
    train_x = torch.randn([num_sample//64, L, dim],device=device).tile(64,1,1)*0.1 + train_x
    print('shape of x: ',train_x.shape)
    
    loss:Tensor = (train_x[:,:-1,:]-train_x[:,1:,:]).square().mean()
    print('directly copy input to output loss(training loss should be lower then this):', loss.item())
    
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5, betas=[0.9,0.98], weight_decay=0.01)
    #warm up
    lr_schedu = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1, total_iters=100)
    
    train_loader = torch.utils.data.DataLoader(train_x, batch_size=Bs, shuffle=True, num_workers=0)
    
    losses = 1
    bar = tqdm(range(500), ncols=80)
    for epoch in bar:
        bar.set_description_str(f'epoch: {epoch}/10000')
        lr_schedu.step()
        for i, x in enumerate(train_loader):
            x:Tensor
            
            out = model.forward(x)
            
            #MSE with target shift
            loss:Tensor = (out[:,:-1,:]-x[:,1:,:]).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses = 0.98*losses + 0.02*loss.item()
            bar.set_postfix_str(f'loss: {losses: .4f}')
        
        
