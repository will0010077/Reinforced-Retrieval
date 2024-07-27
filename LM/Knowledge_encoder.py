import sys
sys.path.append("../")

# Load model directly
import torch

from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
import config

bert_dir = config.bert_dir
enc_size_config = config.enc_size_config
tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
# token = tokenizer.convert_ids_to_tokens(torch.arange(1000)) #998-104
class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers, dims, num_prefix, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        
        self.model = BertModel(BertConfig(**enc_size_config))
        bert = BertModel.from_pretrained(bert_dir)
        self.model.embeddings = bert.embeddings
        self.model.encoder.layer[0] = bert.encoder.layer[0]
        del bert
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.num_layers=num_layers
        self.num_prefix = num_prefix
        self.dims=dims
        self.dtype=dtype
        self.encoder_heads= nn.Linear(enc_size_config['hidden_size'], self.dims)
        
        self.adaption_prompt = nn.Parameter(
            torch.empty(self.num_layers-1, self.num_prefix, self.dims, device=self.model.device, dtype=self.dtype).normal_()
        )#(layer, P, d)
        assert num_prefix<999-104
        self.register_buffer('prefix_tokens', torch.arange(104, 104+self.num_prefix).unsqueeze_(0), persistent=False)#(1,P)
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
        input_ids = torch.cat([self.prefix_tokens.tile([B*k,1]), x.input_ids], dim = 1)[:,:512]
        attention_mask = torch.cat([torch.ones([B*k, self.num_prefix], device = x.input_ids.device), x.attention_mask], dim = 1)[:,:512]


        y=self.model(input_ids =input_ids, attention_mask = attention_mask)
        
        
        y = y.last_hidden_state[:,:self.num_prefix,:]#(B*k, P, d)
        y = self.encoder_heads.forward(y)#(B*k, P, dim)
        # print('encoder_heads output:', y.shape)
        y = y.to(self.dtype)
        y = y.reshape([B, k, self.num_prefix, self.dims, ])#(B, k, P, dim)
        # print('prefix output:', y.shape)
        
        
        # print('before cat output:', y.shape)
        y = y.reshape([B, k*self.num_prefix, self.dims])#(B, k*P, dims)
        y = y.to(device, non_blocking=True)
        # static prefix + Enc prefix
        return self.adaption_prompt.unsqueeze(1).unbind() + (y,)
    
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
    
