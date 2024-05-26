import sys
sys.path.append("../../")

# Load model directly
import torch

from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
import yaml

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
config = config['model_config']

class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers=40, num_heads=40, dims=128, groups=4, num_prefix = 4):
        super().__init__()
        if dims % groups !=0:
            raise ValueError(f'Dims must divided by groups')
        
        self.model = BertModel(BertConfig(num_hidden_layers=2))
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dims=dims
        self.encoder_heads=nn.Conv1d(config['embed_dim']*self.num_layers, self.num_heads*self.dims*self.num_layers, kernel_size=1, groups=groups*self.num_layers)
        
        assert num_prefix<99
        self.num_prefix = num_prefix
        self.register_buffer('prefix_tokens', torch.arange(2, 2+self.num_prefix).unsqueeze_(0))#(1,P)
        self.prefix_tokens:Tensor
    def forward(self, x, k=0, dtype=torch.float32, device = None)->tuple[torch.Tensor, torch.Tensor]:
        '''
        x: (B*k, n)
        output: list[(2, B, head, k*P, dims)]*layer , [B, k*P]
        '''
        if device is None:
            device = self.model.device
        if type(x)==list:
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        
        B=x.input_ids.shape[0]//k
        n=x.input_ids.shape[1]
        x.input_ids = torch.cat([self.prefix_tokens.tile([B*k,1]), x.input_ids], dim = 1)
        x.attention_mask = torch.cat([torch.ones([B*k, self.num_prefix], device = x.input_ids.device), x.attention_mask], dim = 1)
        x = {'input_ids':x.input_ids, 'attention_mask':x.attention_mask}
        y=self.model(**x)
        
        
        y = y.last_hidden_state[:,:self.num_prefix,:]
        pooler_output = y.mean(dim=1)
        y = torch.tile(y,[1,1,self.num_layers])#(B*k, P, in*layer)
        y = y.transpose(1,2)#(B*k, in*layer, P)
        y = self.encoder_heads.forward(y)#(B*k, 5120*layer, P)
        # print('encoder_heads output:', y.shape)
        y = y.to(dtype)
        y = y.reshape([B, k, self.num_layers, -1, self.dims, self.num_heads, self.num_prefix])
        # print('prefix output:', y.shape)
        
        
        y = y.permute([2,3,0,5,1,6,4])#(layer, -1, B, head, k, P, dims)
        # print('before cat output:', y.shape)
        y = y.reshape([self.num_layers, -1, B, self.num_heads, k*self.num_prefix, self.dims])#(layer, 1, B, head, k*P, dims)
        y = y.to(device, non_blocking=True)
        # y = torch.tile(y, [1,2,1,1,1,1])#(layer, 2, B, head, k*P, dims)
        masks = torch.ones([B, k*self.num_prefix], dtype=torch.long, device=y.device)
        return y.unbind(), masks
    
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

if __name__=='__main__':
    model=KnowEncoder()
    # prefix, prefix_masks = model.forward(['hello'], k = 1, dtype=torch.float16)
    output = model.forward(['hello 1','hello 2','hello 3','hello 4','hello 5','hello 6'], 3)
    
