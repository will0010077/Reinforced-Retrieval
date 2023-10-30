import sys
sys.path.append("../../")

# Load model directly
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class KnowEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.num_layer=40
        self.num_heads=40
        self.dim=128
        self.encoder_heads=nn.Conv1d(768*self.num_layer*2, self.num_heads*self.dim*self.num_layer*2, kernel_size=1, groups=32*self.num_layer*2)
    def forward(self, x, k):
        x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        y=self.model(**x)
        B=y[0].shape[0]//k
        n=y[0].shape[1]
        pooler_output=self.mean_pooling( y[0], x['attention_mask'])
        y = torch.tile(y[0],[1,1,self.num_layer*2])#(B*k,n,768*40*2)
        y = y.transpose(1,2)#(B*k,768*40,n)
        y = self.encoder_heads(y)#(B*k,5120*40,n)
        y = y.reshape([B, k, self.num_layer, 2, self.num_heads, self.dim, n])
        y = y.permute([0,1,2,3,4,6,5])#(B, k, 40, 2, 40, n, 128)

        #concat k to n
        batch=[]
        masks=[]
        for i in range(B):
            cat=[]
            for j in range(k):
                masked_feature = y[i,j,...,:x['attention_mask'][i*k+j].sum(dim=0),:]
                cat.append(masked_feature)
            cat = torch.cat(cat, dim=-2)#(40, 2, 40, n*k, 128)
            batch.append(cat.permute(3,0,1,2,4))#(n*k, 40, 2, 40, 128)
            masks.append(torch.ones([cat.shape[-2]], dtype=torch.long))
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True)
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)#(B, n*k, 40, 2, 40, 128)
        batch = batch.permute(0,2,3,4,1,5)#(B, 40, 2, 40, n*k, 128)

        return pooler_output, batch, masks
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

if __name__=='__main__':
    model=KnowEncoder()
    out = model.forward(['hello','hello world!!!!'],k=1)
    print(out[1].shape, out[2].shape, out[2])
