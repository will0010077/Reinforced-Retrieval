import sys
sys.path.append("../../")

# Load model directly
import torch

from transformers import AutoTokenizer, AutoModel
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import pickle
import random
import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
model_config = config['model_config']

seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])

def MSE(a:torch.Tensor, b:torch.Tensor):
    '''a:(B,d), b:(N,d)\\
    out:(B,N)'''
    return torch.mean((a[:,None,:] - b[None,:,:])**2, dim=2)


def check_Qmark(text:str):
    # Reduce sensitivity to question marks
    text=text.replace('？','?')
    while '??' in text:
        text=text.replace('??','?')
    if '?' not in text:
        text+='?'
    return text


class DOC_Retriever(torch.nn.Module):
    def __init__(self, use_cache=True,load_data_feature=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model=Contriever()
        self.model.load_state_dict(torch.load('/home/devil/workspace/nlg_progress/backend/app/save/contriever.pt',map_location='cpu'))
        self.model.eval()
        self.model.to('cuda')
        self.bs=512
        if use_cache:
            self.retrieve_cache= {}
        if load_data_feature:
            self.data=torch.load('/home/devil/workspace/nlg_progress/backend/app/data/data_reduced.pt')
            self.feature=torch.load('/home/devil/workspace/nlg_progress/backend/app/data/vecs_reduced.pt')
    def collate(self, ids):
        ids = torch.stack(ids)
        return {'input_ids':ids.to(self.model.model.device)}#, 'attention_mask':torch.ones_like(ids, dtype=torch.long, device=self.model.model.device)}
    @torch.inference_mode()
    def get_feature(self, ids)->torch.Tensor:
        '''
        return: tensor with shape:(N, 768)
        '''
        feature_list=torch.empty(len(ids),model_config['embed_dim'],dtype=torch.float32)


        dataloader = DataLoader(ids, batch_size=self.bs, shuffle=False, collate_fn=self.collate,num_workers=0)
        for i,idx in (bar:=tqdm(enumerate(dataloader),ncols=0,total=len(dataloader))):
            feature  = self.model(idx)#(bs, d)
            feature_list[i*self.bs:i*self.bs+self.bs]=feature.to('cpu')

        return  feature_list
    def load_index(self):
        '''
        load trained index
        '''
        # with open('/home/devil/workspace/nlg_progress/backend/app/data/IndexIVFFlat.pkl', 'rb') as f:
        #     self.index = pickle.load(f)
        self.index =faiss.read_index(self.index, "/home/devil/workspace/nlg_progress/backend/app/data/accumulated_index.index")

    #TODO
    def train_add_index(self, vectors):
        '''
        vectors:(N,768)\\
        save trained index
        '''
        quantizer=faiss.IndexFlatIP(model_config['embed_dim'])
        self.index = faiss.IndexIVFFlat(quantizer,model_config['embed_dim'],int((10**7)**0.5), faiss.METRIC_INNER_PRODUCT)
        
        assert not self.index.is_trained
        random_sequence = torch.randperm(vectors.shape[0])
        num_samples = config['data_config']['num_doc']
        random_vecs = random_sequence[:num_samples]
        print('training')
        self.index.train(vectors[random_vecs].numpy())
        print('finish training')
        assert self.index.is_trained
        Bs=10**4
        for i in tqdm(range(0, len(vectors), Bs)):
            self.index.add(vectors[:Bs].numpy())
            vectors=vectors[Bs:]


        faiss.write_index(self.index, "/home/devil/workspace/nlg_progress/backend/app/data/accumulated_index.index")

        # f_index=open('/home/devil/workspace/nlg_progress/backend/app/data/IndexIVFFlat.pkl','wb')
        # pickle.dump(self.index,f_index,protocol=4)
        print('saved trained index')
    #TODO
    def search(self, querys, k=1):
        '''
        querys: (B,768)\\
        return: (B, k) index
        '''
        
        self.index.nprobe=10
        D, I = self.index.search(querys, k)
        return I
    def build_index(self, ids, filename):
        self.feature= self.get_feature(ids)
        torch.save(self.feature,filename)
        print('saved vecs_reduced.pt')
    @torch.inference_mode()
    def retrieve(self, querys:list[str], k=5, threshold=0.2):
        '''
        querys: list or str\\
        return retrieved seg (B, k, 512) 
        '''
        if isinstance(querys, str):
            querys = [querys]
            
        if hasattr(self, 'retrieve_cache'):
            #check all query in cache
            if all([q in self.retrieve_cache for q in querys]):
                #check value k greater than current k
                if all([self.retrieve_cache[q].shape[0]>=k for q in querys]):
                    return torch.stack([self.retrieve_cache[q][:k] for q in querys]) #return value in cache
            
            
        with torch.no_grad():
            x=self.tokenizer(querys, return_tensors='pt', padding=True ,truncation=True).to(self.model.model.device)
            query_feature = self.model(x)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        #cosine similarity
        query_feature=query_feature.to(self.feature.device)
        dis = cos_sim(query_feature, self.feature)

        #top-k vector and index
        v, id_batch = torch.topk(dis, k, dim=1, largest=True)
        # print(id_batch.shape)
        id_batch=id_batch.to(self.data.device)
        retrieved_segs = self.data[id_batch]#shape:(B,k,512)
        
        #update retrieve cache
        if hasattr(self, 'retrieve_cache'):
            self.retrieve_cache.update(zip(querys, retrieved_segs)) #shape:(k,512)
            
        return retrieved_segs


class Contriever(torch.nn.Module):
    def __init__(self):
        super(Contriever, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    def forward(self, x:dict):
        '''
        x: dict[input_ids, attention_mask]
        output: Tensor[B, d]
        '''
        mask = x.get('attention_mask',None)
        if mask is None:
            mask = torch.ones_like(x['input_ids'],dtype=torch.long)
        y=self.model(input_ids = x.get('input_ids',None), attention_mask = mask)
        y=self.mean_pooling( y[0], mask)
        return y
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

if __name__=='__main__':
    retriever=DOC_Retriever()
    ooo=retriever.retrieve(['where is taiwan?','DOTDOTDOT'])
    print(ooo.shape)