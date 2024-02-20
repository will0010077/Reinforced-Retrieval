import sys
sys.path.append("../../")

# Load model directly
import torch
from torch import Tensor, nn

from transformers import AutoTokenizer, AutoModel
import logging
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import random
import yaml

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
model_config = config['model_config']

# seed = config['seed']
# torch.manual_seed(seed)
# random.seed(seed)
device='cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
def cos_sim(a:Tensor, b:Tensor):
    '''a:(N,d),b:(M,d)
    out: (N,M)'''
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])

def MSE(a:torch.Tensor, b:torch.Tensor):
    '''a:(N,d),b:(M,d)
    out: (N,M)'''
    return torch.mean((a[:,None,:] - b[None,:,:])**2, dim=2)


def check_Qmark(text:str):
    # Reduce sensitivity to question marks
    text=text.replace('？','?')
    while '??' in text:
        text=text.replace('??','?')
    if '?' not in text:
        text+='?'
    return text

class cluster_builder:
    def __init__(self, num_search=10, k = 3000, bs = 10**5, tol=1, lr=0.2):
        self.centers = None
        self.idx = None
        self.dist_fn = lambda a,b: -cos_sim(a,b)
        self.num_search =  int(num_search)
        self.bs = int(bs)
        self.k = int(k)
        self.tol = tol
        self.lr = lr
        # self.dist_fn = lambda a,b: MSE(a,b)
    def get_mu(self, x, r, mu, a):
        '''x: (n, c), r: (n), mu: (k, c), a: in [0,1]'''
        u = [x[r==i].mean(dim=0) for i in range(self.k)]
        u = torch.stack(u)
        u = u*a + mu*(1-a)
        dis=(u-mu).square().sum(dim=1).mean().item()
        return u, dis
    def get_r(self, x, mu):
        '''x: (n, c), mu: (k, c), r: (n)'''
        dis=self.dist_fn(x, mu)
        min_k=torch.argmin(dis, axis=1)
        return min_k
    
    def init_mu(self, x:Tensor, k):
        '''random select k start point from data to avoid some cluster do not have any data.'''
        perm = torch.randperm(len(x), dtype=torch.int)
        return x[perm[:k]]
    def train(self, data, epoch=10):
        print('cluster training...')
        # assert len(data)>=self.k

        self.data = data
        loader = DataLoader(self.data, batch_size=self.bs, shuffle=True, drop_last=True )
        
        mu = self.init_mu(self.data, self.k).to(device)
        for i in range(epoch):
            bar=  tqdm(loader, ncols = 80)
            for data in bar:
                data = data.to(device)
                dis=float('inf')
                while dis>self.tol:
                    data = torch.cat([data,mu],dim=0)
                    r = self.get_r(data, mu)
                    mu, dis = self.get_mu(data, r, mu, self.lr)
                    bar.set_description_str(f'dist:{dis:.4f}')
        
        self.idx = self.get_idx(mu)
        self.centers=mu
        return self.idx, self.centers
        
    def get_idx(self, mu):
        '''compute each data->cluster
        out: (N)'''
        loader = DataLoader(self.data, batch_size=self.bs, shuffle=False)
        bar=  tqdm(loader, ncols = 80)
        idx = []
        for data in bar:
            data = data.to(device)
            r = self.get_r(data, mu)
            idx.append(r)
        idx = torch.cat(idx).cpu()
        return idx
        
    def build(self, data=None):
        '''build clusters: list[Tensor(n,d)] and idx: list[Tensor(n)]'''
        if self.centers is None:
            raise RuntimeError('The cluster is not trained.')
        print('cluster building...')
        if data is not None:
            self.data = data
            self.idx = self.get_idx(self.centers)
        temp_idx = self.idx
        argsort_idx = temp_idx.argsort()
        _, count = temp_idx.unique(return_counts = True)
        del _
        count=count.tolist()
        series = torch.arange(len(self.idx))
        series = series[argsort_idx]
        self.clusted_idx = series.split(count)
        print('build idx done...')
        
        print('sorting...')
        # def inplace_swap(arr, i, j):
        #     """Function to swap elements in the array 'arr' at indices 'i' and 'j'."""
        #     arr[i], arr[j] = arr[j].clone(), arr[i].clone()
        # order = argsort_idx.argsort()
        # for i in tqdm(range(len(order))):
        #     while  i!= order[i]:
        #         inplace_swap(self.data, i, order[i].item())
        #         inplace_swap(order, i, order[i].item())

        '''this will cause OOM, need to be done in-place'''
        self.data[:] = self.data[argsort_idx]
        '''--------------------------------'''
        self.clusted_data = self.data.split(count)
        del self.data
        print('build cluster done...')
        
        
        del self.idx

        
    def save(self,t):
        '''save clusted_data, center, idx'''
        print('cluster saving...')
        if t is None:
            t = time.time()
        t=int(t)
        data_path = f'./data/clusted_data_{t:d}.pt'
        centers_path = f'./data/centers_{t:d}.pt'
        idx_path = f'./data/idx_{t:d}.pt'
        
        torch.save(self.centers, centers_path)
        print('saving center done...')
        torch.save(self.clusted_idx, idx_path)
        print('saving idx done...')
        torch.save(self.clusted_data, data_path)
        print('saving data done...')
        return True
    
    def load(self,t):
        '''load clusted_data'''
        print('cluster loading...')
        assert t is not None
        t=int(t)
        data_path = f'./data/clusted_data_{t:d}.pt'
        centers_path = f'./data/centers_{t:d}.pt'
        idx_path = f'./data/idx_{t:d}.pt'
        
        self.centers = torch.load(centers_path, map_location='cpu')
        if self.centers.shape[0]!=self.k:
            raise RuntimeError(f'The cluster with k={self.centers.shape[0]} and init k={self.k} are not the same!')
        self.centers=self.centers.to(device)
        print('loading center done...')
        
        self.clusted_idx=torch.load(idx_path)
        print('loading idx done...')
            
        self.clusted_data=torch.load(data_path)
        print('loading data done...')
        return True
    
    def search(self, x, k):
        '''x:(B,c)
        out: (B,k), (B,k,d)
        '''
        
        '''need a function that return true doc idx given clusted idx
        so I need a {idx: doc_idx}'''
        
        idx = self.first(x, k)#(B,k,2)
        ret_idx=[]
        ret_emb=[]
        for top_k in idx:
            b_idx=[]
            b_emb=[]
            for f,s in top_k:
                b_idx.append(self.clusted_idx[f][s])
                b_emb.append(self.clusted_data[f][s])
            b_idx = torch.stack(b_idx)
            b_emb = torch.stack(b_emb)
            ret_idx.append(b_idx)
            ret_emb.append(b_emb)
        ret_idx = torch.stack(ret_idx)
        ret_emb = torch.stack(ret_emb)
        return ret_idx, ret_emb
    
    def first(self, x,k):
        '''x:(B,d)
        out:(B,k,2)'''
        if self.centers is None:
            raise RuntimeError('The cluster is not trained.')
        dist = self.dist_fn(x, self.centers)#(N,k)
        _, c_idx= dist.topk(self.num_search, dim = 1, largest=False)#(N)
        c_idx=c_idx.cpu()
        # print('first:',c_idx)
        
        idx = []
        for i, v in enumerate(x):
            v=v.cpu()
            v_dist = []
            v_idx = []
            for c in c_idx[i]:
                v_c_dist, v_c_idx = self.second(v, self.clusted_data[c], k)
                v_dist.append(v_c_dist)
                v_idx.append(torch.stack([c.tile(len(v_c_idx)), v_c_idx]).T)
            v_dist = torch.cat(v_dist)
            v_idx = torch.cat(v_idx)#(num, k, 2)
            # print('v_idx:',v_idx,v_idx.shape)
            # print('v_dist:',v_dist,v_dist.shape)
            _, v_k_idx = v_dist.topk(k, dim = 0, largest=False)#(k)
            idx.append(v_idx[v_k_idx])
        idx = torch.stack(idx)
        return idx
        
    def second(self,v,c,k):
        '''v:(d), c:(n,d)
        out:(k)'''
        dist = self.dist_fn(v[None,:], c)[0]#(n)
        d, idx = dist.topk(min(k,len(c)), dim = 0, largest=False)#(k)
        return d.cpu(), idx.cpu()
        
    
        
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