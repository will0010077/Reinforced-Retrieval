import sys
sys.path.append("../../")

# Load model directly
import torch

from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
import yaml

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
config = config['Enc_size_config']
bert_dir = "huggingface/bert"
tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
# token = tokenizer.convert_ids_to_tokens(torch.arange(1000)) #998-104
class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers, dims, groups, num_prefix, dtype=torch.float16, **kwargs):
        super().__init__()
        if dims % groups !=0:
            raise ValueError(f'Dims must divided by groups')
        
        self.model = BertModel(BertConfig(**config))
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.num_layers=num_layers
        self.dims=dims
        self.dtype=dtype
        self.encoder_heads=nn.Sequential(
            nn.Conv1d(config['hidden_size'], config['hidden_size']//2, kernel_size=1, groups=groups),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(config['hidden_size']//2, self.dims, kernel_size=1, groups=groups),
        )
        
        assert num_prefix<999-104
        self.num_prefix = num_prefix
        self.register_buffer('prefix_tokens', torch.arange(104, 104+self.num_prefix*num_layers).unsqueeze_(0))#(1,P)
        self.prefix_tokens:Tensor
    def forward(self, x, k=1, device = None)->tuple[torch.Tensor]:
        '''
        x: (B*k, n)
        output: (layer, B, k*P, dims)
        '''
        if device is None:
            device = self.model.device
        if isinstance(x,list):
            assert isinstance(x[0], str)
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        
        B=x.input_ids.shape[0]//k # bs
        n=x.input_ids.shape[1] # length


        # cat special token for output fixed length of embedding, update attention mask length
        x.input_ids = torch.cat([self.prefix_tokens.tile([B*k,1]), x.input_ids], dim = 1)
        x.attention_mask = torch.cat([torch.ones([B*k, self.num_prefix*self.num_layers], device = x.input_ids.device), x.attention_mask], dim = 1)


        y=self.model(input_ids =x.input_ids, attention_mask = x.attention_mask)
        
        
        y = y.last_hidden_state[:,:self.num_prefix*self.num_layers,:]#(B*k, P*layer, d)
        y = y.transpose(1,2)#(B*k, d, P*layer)
        y = self.encoder_heads.forward(y)#(B*k, dims, P*layer)
        # print('encoder_heads output:', y.shape)
        y = y.to(self.dtype)
        y = y.reshape([B, k, self.dims, self.num_prefix, self.num_layers, ])#(B, k, dims, P, layer)
        # print('prefix output:', y.shape)
        
        
        y = y.permute([4,0,1,3,2])#(layer, B, k, P, dims)
        # print('before cat output:', y.shape)
        y = y.reshape([self.num_layers, B, k*self.num_prefix, self.dims])#(layer, B, k*P, dims)
        y = y.to(device, non_blocking=True)
        return y.unbind()
    
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

if __name__=='__main__':
    model=KnowEncoder(4,4096,8,16)
    print(model.model.training)
    # prefix, prefix_masks = model.forward(['hello'], k = 1, dtype=torch.float16)
    output = model.forward(['hello 1','hello 2','hello 3','hello 4','hello 5','hello 6'], 3)
    print([output[i].shape for i in range(len(output))])
    
