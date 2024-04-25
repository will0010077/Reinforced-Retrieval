import sys
sys.path.append("..")

# Load model directly
import torch
from torch import Tensor, nn
from DocBuilder.utils import inner, collate_list_to_tensor, custom_sparse_mmT, top_k_sparse

from transformers import AutoTokenizer, AutoModel
import logging
import time, datetime
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

class cluster_builder:
    def __init__(self, k = 3000, sparse_dim=128):
        self.centers = None
        self.idx = None
        self.dim = None
        self.k = int(k)
        self.sparse_dim = sparse_dim
    def get_mu(self, x:Tensor, r:Tensor, mu:Tensor, lr:float):
        '''x: (n, c), r: (n), mu: (k, c), a: in [0,1]'''
        u = [x[r==i].mean(dim=0) for i in range(self.k)]
        u = torch.stack(u)
        u = u*lr + mu*(1-lr)
        u = top_k_sparse(u, 512).to_dense()

        dis=(u-mu).norm(dim=-1).mean()

        return u, dis

    def get_r(self, x, mu):
        '''x: (n, c), mu: (k, c), r: (n)'''
        dis=inner(x, mu)#this was a bug!! please rename them
        min_k=torch.argmax(dis, axis=1)
        min_k[-self.k:]=torch.arange(self.k, device=min_k.device)
        return min_k

    def select_init_mu(self, x, k):
        '''random select k start point from data to avoid some cluster do not have any data.'''
        perm = torch.randperm(len(x), dtype=torch.int)
        # return x[:k]
        return torch.stack([x[i] for i in perm[:k]]).to_dense()
    def rand_init_mu(self, x, k):
        '''random start point '''
        return torch.randn((k,len(x[0])))

    def train(self, data, epoch=10, bs = 10**5, tol=0.1, lr=0.2):
        print('cluster training...')
        # assert len(data)>=self.k

        self.data = data
        loader = DataLoader(self.data, batch_size=bs, shuffle=True, collate_fn=collate_list_to_tensor, num_workers=4, persistent_workers=True)

        mu = self.select_init_mu(self.data, self.k).to(device)
        for _ in range(epoch):
            bar=  tqdm(loader, ncols = 80)
            for data in bar:
                data:Tensor = data.to(device)
                data.sparse_resize_([data.shape[0]+self.k, data.shape[1]], data.sparse_dim(), data.dense_dim())
                data = data.to_dense()
                data[-self.k:] = mu
                dis=float('inf')
                count=0
                while dis>tol and count<100:
                    count+=1
                    # data[-self.k:, :]=mu
                    r = self.get_r(data, mu)
                    mu, dis = self.get_mu(data, r, mu, lr)
                    bar.set_description_str(f'dist:{dis:.4f}')
            del data, r
        del loader
        
        self.idx = self.get_idx(mu, bs).cpu()
        self.centers=mu
    
        return self.idx, self.centers

    def get_idx(self, mu, bs):
        '''compute each data->cluster
        out: (N)'''
        loader = DataLoader(self.data, batch_size=bs, shuffle=False, collate_fn=collate_list_to_tensor, num_workers=4)
        bar=  tqdm(loader, ncols = 80)
        idx = []
        for data in bar:
            data:Tensor = data.to(device)
            data = data.to_dense()
            r = self.get_r(data, mu)
            idx.append(r)
        idx = torch.cat(idx)
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

        '''this will cause OOM, need to be done in-place'''
        # self.data[:] = self.data[argsort_idx]
        '''--------------------------------'''
        '''new sort method'''
        self.data = sorted(zip(self.data, argsort_idx.argsort()), key=lambda x:x[1])
        self.data = list(zip(*self.data))[0]
        '''-----------------------'''
        # self.clusted_data = self.data.split(count)
        '''new split method for list'''
        self.clusted_data = [torch.stack(self.data[sum(count[:i]):sum(count[:i+1])]).coalesce() for i in range(len(count))]
        print('coalesced:',self.clusted_data[0].is_coalesced())
        del self.data
        print('build cluster done...')


        del self.idx


    def save(self,name=None):
        '''save clusted_data, center, idx'''
        print('cluster saving...')
        if name is None:
            name = datetime.datetime.now().strftime('%m_%d_%H_%M')
        data_path = f'data/clusted_data_{name}.pt'
        torch.save({'center':self.centers, 'idx':self.clusted_idx, 'data':self.clusted_data}, data_path)
        print('save done!!')
        return name

    def load(self,name):
        '''load clusted_data'''
        print('cluster loading...')
        assert name is not None
        data_path = f'data/clusted_data_{name}.pt'
        loaded_dict = torch.load(data_path, map_location='cpu')
        self.centers = loaded_dict['center']
        self.clusted_idx = loaded_dict['idx']
        self.clusted_data = loaded_dict['data']
        print('coalesced:',self.clusted_data[0].is_coalesced())
        self.dim = self.centers.shape[-1]
        if self.centers.shape[0]!=self.k:
            raise RuntimeError(f'The cluster with k={self.centers.shape[0]} and init k={self.k} are not the same!')
        # self.centers=self.centers.to(device)
        print('load done!!')
        return True

    def search(self, x, k, num_search=10):
        '''x:(B,c)
        out: (B,k), (B,k,d)
        '''

        '''need a function that return true doc idx given clusted idx
        so I need a {idx: doc_idx}'''

        idx = self.first(x, k, num_search)#(B,k,2)
        ret_idx=[]
        ret_emb=[]
        for top_k in idx:
            b_idx=[]
            b_emb=[]
            for f,s in top_k:
                b_idx.append(self.clusted_idx[f][s])
                # b_emb.append(self.clusted_data[f][s])# RuntimeError: Cannot set version_counter for inference tensor
                mask = self.clusted_data[f].indices()[0]==s
                indices = self.clusted_data[f].indices()[1][mask].unsqueeze(0)
                values = self.clusted_data[f].values()[mask]
                v = torch.sparse_coo_tensor(indices, values, [self.dim]).to_dense()
                b_emb.append(v)
            b_idx = torch.stack(b_idx)
            b_emb = torch.stack(b_emb)
            ret_idx.append(b_idx)
            ret_emb.append(b_emb)
        ret_idx = torch.stack(ret_idx)
        ret_emb = torch.stack(ret_emb)
        return ret_idx, ret_emb

    def first(self, x, k, num_search):
        '''x: query (B,d)
        out: outptu idxs (B,k,2) (cluster_id, top_id)'''
        if self.centers is None:
            raise RuntimeError('The cluster is not trained.')
        dist = inner(x, self.centers)#(N,k)
        _, c_idx= dist.topk(num_search, dim = 1)#(N)
        c_idx=c_idx.cpu()
        # print('first:',c_idx)

        idx = []
        for i, v in enumerate(x):
            v = top_k_sparse(v[None,:], self.sparse_dim)[0].coalesce()
            v=v.cpu()
            
            v_dist = []
            v_idx = []
            for c in c_idx[i]:
                # original inner product
                # v_c_dist, v_c_idx = self.second(v, self.clusted_data[c].to_dense(), k) # 15 iter/sec
                # new operation for sparse tensor
                v_c_dist, v_c_idx = self.second(v, self.clusted_data[c], k)  # 50 iter/sec
                
                v_dist.append(v_c_dist)
                v_idx.append(torch.stack([c.tile(len(v_c_idx)), v_c_idx]).T)
            v_dist = torch.cat(v_dist)
            v_idx = torch.cat(v_idx)#(num, k, 2) (cluster_id, top_id)
            # print('v_idx:',v_idx,v_idx.shape)
            # print('v_dist:',v_dist,v_dist.shape)
            _, v_k_idx = v_dist.topk(k, dim = 0)#(k)
            idx.append(v_idx[v_k_idx])
        idx = torch.stack(idx)
        return idx

    def second(self, v ,c ,k):
        '''v: query with shape (d)
        c: cluster vectors with shape(n,d)
        out: output idx() with shape (k)'''
        if c.is_sparse:
            dist = custom_sparse_mmT(v, c)
        else:
            dist = inner(v[None,:], c)[0]#(n)
        d, idx = dist.topk(min(k,len(c)), dim = 0)#(k)
        return d.cpu(), idx.cpu()



class doc_retriever(torch.nn.Module):
    def __init__(self, model:nn.Module, data:Tensor, cluster:cluster_builder, use_cache=True, **kargs):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.model=model
        self.model.eval()
        self.data= data
        self.cluster = cluster
        self.ref = torch.nn.Parameter(torch.zeros(1))
        if use_cache:
            self.retrieve_cache= {}

    @torch.inference_mode()
    def retrieve(self, querys:list[str], k=5, num_search=10) ->tuple[Tensor, Tensor] :
        '''
        querys: list or str or Tensor\\
        return retrieved seg (B, k, len), embbeding (B,k,d)
        '''
        if isinstance(querys, str):
            querys = [querys]
            
        if isinstance(querys, list):
            query_feature = self.forward(querys)
        elif isinstance(querys, Tensor):
            query_feature = querys
        else:
            raise TypeError(querys)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        idx, emb = self.cluster.search(query_feature.to(self.cluster.centers.device), k, num_search)
        #cosine similarity
        idx=idx.to(self.data.device)
        retrieved_segs = self.data[idx]#shape:(B,k,len)


        return retrieved_segs, emb
    
    def forward(self, querys:list[str]):
        x=self.tokenizer(querys, return_tensors='pt', padding=True ,truncation=True).to(self.ref.device)
        return  self.model(x)
        
if __name__=='__main__':
    retriever=doc_retriever()
    ooo=retriever.retrieve(['where is taiwan?','DOTDOTDOT'])
    print(ooo.shape)
