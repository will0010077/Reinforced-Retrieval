import sys
sys.path.append("../")

# Load model directly
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel, RobertaConfig, RobertaModel
import config

bert_dir = config.bert_dir
enc_size_config = config.enc_size_config
tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
# token = tokenizer.convert_ids_to_tokens(torch.arange(1000)) #998-104
class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers, dims, num_prefix, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        
        self.model = RobertaModel(RobertaConfig(**enc_size_config))
        bert = RobertaModel.from_pretrained(config.roberta_dir)
        self.model.embeddings = bert.embeddings
        for i in range(enc_size_config.num_hidden_layers):
            self.model.encoder.layer[i] = bert.encoder.layer[i]
        del bert
        self.model.requires_grad_(True)
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(config.roberta_dir)
        self.num_layers=num_layers
        self.num_prefix = num_prefix
        self.dims=dims
        self.dtype=dtype
        self.num_head = 32
        # Linear layer to project the output of the attention layer to the desired dimension
        self.proj_layer = nn.Linear(enc_size_config['hidden_size'], self.dims)
        self.attention_pooling = nn.MultiheadAttention(self.dims, self.num_head, 0., batch_first=True)
        # Adaption prompt as before
        self.adaption_prompt = nn.Parameter(
            torch.empty(self.num_layers, self.num_prefix, self.dims, device=self.model.device, dtype=torch.float32).normal_()
        )#(layer, P, d)
        self.enc_len = 32
        self.pooling_drop = 0.
        self.adaption_querys = nn.Parameter(
            torch.empty(1,self.enc_len, self.dims, device=self.model.device, dtype=torch.float32).normal_()
        )

    @property
    def device(self):
        return self.model.device
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x, k=1, device = None, stage = 1)->tuple[torch.Tensor]:
        '''
        x: (B*k, n)
        output: (layer, B, k*P, dims)
        '''
        if device is None:
            device = self.device
        if stage==0:
            return  (None,)+ (None,)*(32-self.num_layers-1) + self.adaption_prompt.to(self.dtype).to(device, non_blocking=True).unsqueeze(1).unbind()
        if isinstance(x,list):
            assert isinstance(x[0], str)
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        
        B=x.input_ids.shape[0]//k # bs
        n=x.input_ids.shape[1] # length


        # cat special token for output fixed length of embedding, update attention mask length
        input_ids = x.input_ids[:,:512]
        attention_mask = x.attention_mask[:,:512]


        y=self.model(input_ids =input_ids, attention_mask = attention_mask)
        
        
        y = y.last_hidden_state#(B*k, N, d)
        y = self.proj_layer.forward(y)#(B*k, N, dim)


        # attention_mask  = ~attention_mask[:,None,:].bool()# (B,N) -> (B, LQ, N)
        # attention_mask = attention_mask.repeat([self.num_head, self.querys.shape[-2],1])
        # y = self.attention_pooling.forward(self.querys.repeat([B*k,1,1]), y, y, attn_mask = attention_mask, need_weights=False)[0]

        attention_mask  = attention_mask[:,None,:].bool()# (B,N) -> (B, LQ, N)
        y = F.scaled_dot_product_attention(self.adaption_querys, y, y, attn_mask = attention_mask,  dropout_p=(self.pooling_drop if self.training else 0.0)) #(Bk, Qlen, dim)


        # print('encoder_heads output:', y.shape)
        y = y.reshape([B, k, self.enc_len, self.dims, ])#(B, k, P, dim)
        # print('prefix output:', y.shape)
        
        
        # print('before cat output:', y.shape)
        y = y.reshape([B, k*self.enc_len, self.dims])#(B, k*P, dims)
        y = y.to(self.dtype)
        y = y.to(device, non_blocking=True)
        # static prefix + Enc prefix
        # return  (y,)+ (None,)*(32-1) 
        return  (y,)+ (None,)*(32-self.num_layers-1) + self.adaption_prompt.to(self.dtype).to(device).unsqueeze(1).unbind()
    
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


class KnowEncoder_Embedding(torch.nn.Module):
    def __init__(self, num_layers, dims, num_prefix, embedding, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        
        # r = 16
        # self.A = nn.Sequential(nn.Linear(dims, r), nn.LeakyReLU())
        # self.B = nn.Linear(r, dims)
        # self.B.weight.data.zero_()
        
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim, embedding.padding_idx, _weight = embedding.weight.data.clone())
        self.embedding.requires_grad_(True)
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(config.LM_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_layers=num_layers
        self.num_prefix = num_prefix
        self.dims=dims
        self.dtype=dtype
        self.num_head = 32
        # Linear layer to project the output of the attention layer to the desired dimension
        self.proj_layer = nn.Linear(enc_size_config['hidden_size'], self.dims)
        self.attention_pooling = nn.MultiheadAttention(self.dims, self.num_head, 0., batch_first=True)
        # Adaption prompt as before
        self.adaption_prompt = nn.Parameter(
            torch.empty(self.num_layers, self.num_prefix, self.dims, dtype=torch.float32).normal_()
        )#(layer, P, d)
        self.enc_len = 32
        self.pooling_drop = 0.1
        self.adaption_querys = nn.Parameter(
            torch.empty(1,self.enc_len, self.dims, dtype=torch.float32).normal_()
        )

    @property
    def device(self):
        return self.embedding.weight.device
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x, k=1, device = None, stage = 1)->tuple[torch.Tensor]:
        '''
        x: (B*k, n)
        output: (layer, B, k*P, dims)
        '''
        if device is None:
            device = self.device
        if stage==0:
            return  (None,)+ (None,)*(32-self.num_layers-1) + self.adaption_prompt.to(self.dtype).to(device, non_blocking=True).unsqueeze(1).unbind()
        if isinstance(x,list):
            assert isinstance(x[0], str)
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.device)
        
        B=x.input_ids.shape[0]//k # bs
        n=x.input_ids.shape[1] # length


        # cat special token for output fixed length of embedding, update attention mask length
        input_ids = x.input_ids[:,:512]
        attention_mask = x.attention_mask[:,:512]

        y = self.embedding(input_ids).to(self.adaption_querys.dtype)


        # attention_mask  = ~attention_mask[:,None,:].bool()# (B,N) -> (B, LQ, N)
        # attention_mask = attention_mask.repeat([self.num_head, self.querys.shape[-2],1])
        # y = self.attention_pooling.forward(self.querys.repeat([B*k,1,1]), y, y, attn_mask = attention_mask, need_weights=False)[0]

        attention_mask  = attention_mask[:,None,:].bool()# (B,N) -> (B, LQ, N)
        y = F.scaled_dot_product_attention(self.adaption_querys, y, y, attn_mask = attention_mask,  dropout_p=(self.pooling_drop if self.training else 0.0)) #(Bk, Qlen, dim)

        # print('encoder_heads output:', y.shape)
        y = y.reshape([B, k, self.enc_len, self.dims, ])#(B, k, P, dim)
        # print('prefix output:', y.shape)
        
        
        # print('before cat output:', y.shape)
        y = y.reshape([B, k*self.enc_len, self.dims])#(B, k*P, dims)
        y = y.to(self.dtype)
        y = y.to(device, non_blocking=True)
        # static prefix + Enc prefix
        # return  (y,)+ (None,)*(32-1) 
        return  (y,)+ (None,)*(32-self.num_layers-1) + self.adaption_prompt.to(self.dtype).to(device).unsqueeze(1).unbind()
    
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
    
