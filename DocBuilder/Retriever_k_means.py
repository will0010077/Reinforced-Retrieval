import sys
sys.path.append("..")

# Load model directly
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from DocBuilder.utils import inner, collate_list_to_tensor, custom_sparse_mmT, top_k_sparse
from config import cluster_config
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
import logging
import time, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import random
import yaml


device='cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'

class cluster_builder(nn.Module):
    def __init__(self, data = None, k = 3000, sparse_dim=128):
        super().__init__()
        self.centers = None
        self.idx = None
        self.dim = None
        self.k = int(k)
        self.sparse_dim = sparse_dim
        self.data = None
    def get_mu(self, x:Tensor, r:Tensor, mu:Tensor, lr:float):
        '''x: (n, c), r: (n), mu: (k, c), a: in [0,1]'''
        u = []
        for i in range(self.k):
            temp = x[r==i]
            if len(temp)>1:
                u.append(temp[:-1].mean(dim=0))
            else:
                u.append(temp.mean(dim=0))
        u = torch.stack(u)
        u = u*lr + mu*(1-lr)
        u = F.normalize(u, dim=-1)
        u = top_k_sparse(u, cluster_config.k_sparse).to_dense()

        dis=(u-mu).norm(dim=-1).max()

        return u, dis

    def get_r(self, x, mu)->Tensor:
        '''x: (n, c), mu: (k, c), r: (n)'''
        sim=inner(x, mu)
        min_k = torch.argmax(sim, axis=1)
        return min_k

    def select_init_mu(self, x, k)->Tensor:
        '''random select k start point from data to avoid some cluster do not have any data.'''
        perm = torch.randperm(len(x), dtype=torch.int)
        # return x[:k]
        return torch.stack([x[i] for i in perm[:k]]).to_dense()
    def rand_init_mu(self, x, k)->Tensor:
        '''random start point '''
        return torch.randn((k,x[0].shape[0])).relu_()

    def train(self, data, epoch=10, bs = 10**5, tol=0.1, lr=0.2):
        print('cluster training...')
        # assert len(data)>=self.k
        # try this https://arxiv.org/abs/1507.05910
        self.data = data
        self.size = [len(data), len(data[0])]
        loader = DataLoader(self.data, batch_size=bs, shuffle=True, collate_fn=collate_list_to_tensor, num_workers=1, persistent_workers=True)

        mu = self.select_init_mu(self.data, self.k).to(device)
        count_new= 0
        count=[1]
        for _ in range(epoch):
            bar=  tqdm(loader, ncols = 0)
            for data in bar:
                data:Tensor = data.to(device)
                # data.resize_([data.shape[0]+self.k, data.shape[1]])
                data.sparse_resize_([data.shape[0]+self.k, data.shape[1]], data.sparse_dim(), data.dense_dim())
                data = data.to_dense()
                data[-self.k:] = mu
                dis=float('inf')
                it=0
                while dis>tol and it<10:
                    it+=1
                    count_new+=1
                    # data[-self.k:, :]=mu
                    r = self.get_r(data, mu)
                    r[-self.k:]=torch.arange(self.k, device=r.device)
                    mu, dis = self.get_mu(data, r, mu, lr)
                    ele, count = r.unique(return_counts = True)
                    count:Tensor
                    bar.set_description_str(f'dist:{dis:.4f}, max/min: {max(count)/min(count):.1f}')
                    if max(count)/min(count)>20:
                        mu[ele[count.topk(k=5, largest=False).indices]] = mu[ele[count.topk(k=5).indices]]+0.001*torch.randn([5,self.size[1]], device=mu.device)
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
            self.idx = self.get_idx(self.centers, 50000)
        temp_idx = self.idx
        argsort_idx = temp_idx.argsort()
        idx, count = temp_idx.unique(return_counts = True)
        z =  torch.zeros([self.k], dtype=torch.long)
        z[idx] = count
        self.centers = self.centers[z!=0]
        z = z[z!=0]

        count = z.tolist()
        sort_count = sorted(count)
        print('Maximum cluster:',sort_count[-10:],', minimum cluster:',sort_count[:10], 'All:', count)
        
        series = torch.arange(len(self.idx))
        series = series[argsort_idx]
        self.clusted_idx = series.split(count)
        print('build idx done...')

        print('sorting...')

        '''this will cause OOM, need to be done in-place'''
        # self.data[:] = self.data[argsort_idx]
        '''--------------------------------'''
        '''new sort method'''
        self.data = [*zip(self.data, argsort_idx.argsort())]
        self.data.sort(key=lambda x:x[1])
        self.data = [*zip(*self.data)][0]
        '''-----------------------'''
        # self.clusted_data = self.data.split(count)
        '''new split method for list'''
        self.clusted_data = [torch.stack(self.data[sum(count[:i]):sum(count[:i+1])]) for i in range(len(count))]
        # self.clusted_data = [torch.stack(self.data[sum(count[:i]):sum(count[:i+1])]).coalesce() for i in range(len(count))]

        del self.data
        print('build cluster done...')


        del self.idx


    def save(self,name=None):
        '''save clusted_data, center, idx'''
        print('cluster saving...')
        if name is None:
            name = datetime.datetime.now().strftime('%m_%d_%H_%M')
        data_path = f'data/clusted_data_{name}.pt'
        torch.save({'centers':self.centers.to_sparse(), 'idx':self.clusted_idx, 'data':self.clusted_data}, data_path)
        print('save done!!')
        return name

    def load(self,name):
        '''load clusted_data'''
        print('cluster loading...')
        assert name is not None
        data_path = f'data/clusted_data_{name}.pt'
        loaded_dict = torch.load(data_path, map_location='cpu')
        del self.centers
        self.register_buffer('centers', loaded_dict['centers'].to_dense())
        self.centers:Tensor
        sparse_centers = top_k_sparse(self.centers, 32)
        
        self.clusted_idx = loaded_dict['idx']
        
        
        self.clusted_data = loaded_dict['data']
        self.clusted_data = [d.coalesce() if not d.is_coalesced() else d for d in self.clusted_data]
        self.lens = torch.tensor([len(d) for d in self.clusted_data])
        max_cluster=torch.topk(self.lens, 5)
        min_cluster=torch.topk(self.lens, 5, largest=False)
        
        # tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        # print('cluster min:')
        
        # for m in min_cluster.indices:
        #     z = sparse_centers[m]
        #     print(self.clusted_idx[m])
        #     for i, v in sorted(zip(tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
        #         print(f'{i}:{v:.3f}, ',end='')
        #     print()
        # print('max:')
        # for m in max_cluster.indices:
        #     z = sparse_centers[m]
        #     for i, v in sorted(zip(tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
        #         print(f'{i}:{v:.3f}, ',end='')
        #     print()
        print('cluster min:', min_cluster.values, ', max:', max_cluster.values, ', sum:', sum(self.lens))
        
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
            v:Tensor = v.to_sparse()
            v_dist = []
            v_idx = []
            
            sended_data = [self.clusted_data[d].to(self.centers.device, non_blocking=True) for d in c_idx[i]]
            for i, c in enumerate(c_idx[i]):
                # original inner 
                # v_c_dist, v_c_idx = self.second(v, self.clusted_data[c].to_dense(), k) # 15 iter/sec
                # new operation for sparse tensor
                v_c_dist, v_c_idx = self.second(v, sended_data[i], k)  # 50 iter/sec
                
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

    def second(self, v:Tensor ,c:Tensor ,k:int):
        '''v: query with shape (d)
        c: cluster vectors with shape(n,d)
        out: output idx() with shape (k)'''
        assert len(v.shape) == 1
        assert len(c.shape) == 2
        if c.is_sparse:
            dist = custom_sparse_mmT(v, c)
        else:
            dist = inner(v.to_dense()[None,:], c)[0]#(n)
        d, idx = dist.topk(min(k,len(c)), dim = 0)#(k)
        return d.cpu(), idx.cpu()



class doc_retriever(torch.nn.Module):
    def __init__(self, model:nn.Module, data:Tensor, cluster:cluster_builder, use_cache=True, **kargs):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.model=model
        self.model.eval()
        self.model.requires_grad_(False)
        self.data= data
        self.cluster = cluster
        if len(self.data)!=sum(self.cluster.lens):
            raise ValueError(f'number of segments: {len(self.data)} is not equal to cluster: {sum(self.cluster.lens)}')
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
            
        if isinstance(querys[0], str):
            query_feature = self.forward(querys)
        elif isinstance(querys, Tensor):
            query_feature = querys
        else:
            raise TypeError(querys)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        query_feature = query_feature.to(self.cluster.centers.device)
        query_feature = top_k_sparse(query_feature, self.cluster.sparse_dim).to_dense()
        idx, emb = self.cluster.search(query_feature, k, num_search)
        #cosine similarity
        idx=idx.to(self.data.device)
        retrieved_segs = self.data[idx]#shape:(B,k,len)


        return retrieved_segs, emb
    
    def forward(self, querys:list[str]):
        x=self.tokenizer(querys, return_tensors='pt', padding=True ,truncation=True).to(self.cluster.centers.device)
        return  F.normalize(self.model(x), dim=-1)
    
    @property
    def device(self):
        return self.model.device
        
if __name__=='__main__':
    retriever=doc_retriever()
    ooo=retriever.retrieve(['where is taiwan?','DOTDOTDOT'])
    print(ooo.shape)
