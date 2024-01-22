import sys
sys.path.append("../../")

# Load model directly
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import yaml
with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
config = config['model_config']

class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers=40, num_heads=40, dims=128, groups=4):
        super().__init__()
        if dims % groups !=0:
            raise ValueError(f'Dims must divided by groups')
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dims=dims
        self.encoder_heads=nn.Conv1d(config['embed_dim']*self.num_layers*2, self.num_heads*self.dims*self.num_layers*2, kernel_size=1, groups=groups*self.num_layers*2)
    
    def forward(self, x, k=0, dtype=torch.float32)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        # print(len(x))
        # if k==0:
        #     k=len(x)
        # assert len(x)%k==0
        # print(type(x))
        if type(x)==str:
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        y=self.model(input_ids = x.get('input_ids',None), attention_mask = x.get('attention_mask',None))
        B=y[0].shape[0]//k
        n=y[0].shape[1]
        pooler_output=self.mean_pooling( y[0], x['attention_mask'])
        y = torch.tile(y[0],[1,1,self.num_layers*2])#(B*k,n,768*40*2)
        y = y.transpose(1,2)#(B*k,768*40,n)
        y = self.encoder_heads(y)#(B*k,5120*40,n)
        y=y.to(dtype)
        y = y.reshape([B, k, self.num_layers, 2, self.dims, self.num_heads, n])
        y = y.permute([0,1,2,3,5,6,4])#(B, k, 40, 2, 40, n, 128)

        #concat k to n
        batch=[]
        masks=[]
        for i in range(B):
            cat=[]
            for j in range(k):
                masked_feature = y[i,j,...,:x['attention_mask'][i*k+j].sum(dim=0),:]
                cat.append(masked_feature)
            cat = torch.cat(cat, dim=-2)#(40, 2, 40, n*k, 128)
            cat = cat.permute(3,0,1,2,4)#(n*k, 40, 2, 40, 128)
            batch.append(cat)
            masks.append(torch.ones([cat.shape[0]], dtype=torch.long, device=cat.device))
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True)#(B, n*k)
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)#(B, n*k, 40, 2, 40, 128)
        batch = batch.permute(2,3,0,4,1,5)#(40, 2, B, 40, n*k, 128)

        return pooler_output.to(dtype), batch, masks
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

if __name__=='__main__':
    model=KnowEncoder()
    
