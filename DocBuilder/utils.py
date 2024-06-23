import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import multiprocessing as mp
import time
from collections import UserDict
def top_k_sparse(x:Tensor, k:int, vec_dim:int=-1):
    '''
    x: Tensor
    vec_dim: data dim, default -1
    out: sparsed x
    '''
    dim = len(x.shape)
    scale=dim*2+1
    if scale>(x.shape[vec_dim]/k):
        print(f'Warning! Sparsed result larger than original Tensor. scale: {scale}, sparsity: {(x.shape[vec_dim]/k)}')
    assert k<=x.shape[vec_dim]# check k smaller than original size
    return_topk = x.topk(k, dim=vec_dim)
    if dim==2:
        i = torch.stack([torch.arange(x.shape[0], device=x.device).repeat_interleave(k), return_topk.indices.reshape([-1])])
    else:
        i = return_topk.indices.reshape([1,-1])
        
    v = return_topk.values.reshape([-1])
    mask = v!=0
    i, v = i[mask[None,...].repeat([dim,1])].reshape([dim,-1]), v[mask]
    new_x = torch.sparse_coo_tensor(i, v, x.shape, is_coalesced=True)

    # old method
    # a, _=x.argsort(dim=vec_dim).split_with_sizes(split_sizes=[x.shape[vec_dim]-k, k], dim=vec_dim) #keep top k index
    # x.scatter_(dim=vec_dim, index=a, value=0)#other index full with zero
    # x=(x).to_sparse()
    # print((new_x.coalesce().values()==x.coalesce().values()).float().mean())
    return new_x

def generate_mask(x:Tensor, pad:int):
    '''
    x:(B,N) with pad
    output: mask extend one token
    '''
    mask = (x!=pad).long()
    mask:Tensor
    front = torch.ones([len(mask),1], dtype=torch.long, device=mask.device)
    mask = torch.cat([front, mask], dim=-1)[:,:-1]
    return mask

def sparse_retrieve_rep(x:Tensor):
    return torch.log(1+torch.relu_(x))

def max_pooling(token_embeddings:Tensor, mask:Tensor):
    token_embeddings.masked_fill_(~mask.bool()[..., None], float('-inf'))
    sentence_embeddings = torch.max(token_embeddings, dim=1)
    return sentence_embeddings.values

def cos_sim(a:Tensor, b:Tensor):
    '''a:(N,d),b:(M,d)
    out: (N,M)'''
    a = F.normalize(a, dim = -1)
    b = F.normalize(b, dim = -1)
    return inner(a,b)

def inner(a:Tensor, b:Tensor):
    '''a:(N,d),b:(M,d)
    out: (N,M)'''
    similarity = a@b.T
    return similarity

def sparse_inner(a:Tensor, b:Tensor):
    '''a:(N,d),b:(M,d)
    out: (N,M)'''
    return a@b.T

def custom_sparse_mmT(a: Tensor, b: Tensor) -> Tensor:
    '''a: sparse vector shape(d)
    b: sparse matrix shape(M,d)
    output a@b.T with shape (M)
    '''
    
    assert len(a.shape) == 1
    assert len(b.shape) == 2
    # Get indices with values of a
    indices_a = a.indices().squeeze()
    values_a = a.values()
    # Get values of sparse vector a as a dense vector
    a_dense = values_a
    # print(indices_a, a_dense)#check OK
    
    d = a.shape[0]
    k = a_dense.size(0)
    # Get indices with values of b
    indices_b = b.indices()
    values_b = b.values()
    
    # Filter indices of b that are also present in a
    mask = torch.isin(indices_b[1], indices_a)
    filtered_indices_b = indices_b[:, mask]
    filtered_values_b = values_b[mask]
    # print(filtered_indices_b, filtered_values_b)#check OK
    M = b.size(0)
    
    # Create a map from filtered indices of a to new indices [0, k]
    inverted_index = torch.full([d], d, dtype=torch.long, device = b.device)
    inverted_index[indices_a]=torch.arange(k, dtype=torch.long, device = b.device)
    # print(inverted_index)#check OK
    
    # Map the filtered indices of b to new indices [0, k]
    # mapped_indices_b = torch.tensor([[idx[0], inverted_index[idx[1].item()]] for idx in filtered_indices_b.T]).T
    mapped_indices_b = filtered_indices_b
    mapped_indices_b[1] = inverted_index[mapped_indices_b[1]]
    # print(mapped_indices_b)#check OK
    
    # Create a dense matrix from mapped indices and values of b
    # b_dense = torch.sparse_coo_tensor(mapped_indices_b, filtered_values_b, (M, k)).to_dense()
    b_dense = torch.zeros([M,k], dtype=b.dtype, device = b.device)
    b_dense[mapped_indices_b[0], mapped_indices_b[1]] = filtered_values_b

    # Perform matrix multiplication (transposed)
    result_dense = a_dense@b_dense.T
    
    return result_dense
    

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

def collate_list_to_tensor(batch:list[Tensor]):    
    return torch.stack(batch)

def split_list_to_batch(data:list[Tensor], bs = 2**10):
    '''
    data : list of vector
    return : list of batched vector (matrix)
    '''
    size = len(data)
    return [torch.stack(data[i:i+bs]) for i in tqdm(range(0, size, bs), ncols=0)]

def restore_batched_list(data:list[Tensor]):
    '''
    data : list of batched vector (matrix)
    return : list of vector
    '''
    new_data = []
    for _ in tqdm(range(len(data)), ncols=0):
        new_data.extend(data.pop(0))
    return new_data


class unbind_sparse:
    def __init__(self, data:Tensor):
        self.data = data
    def run(self, use_sort=False):
        '''
        2D sparse tensor -> list[1D sparse tensor]
        '''
        self.data = self.data.coalesce()
        size = self.data.shape
        if use_sort:# checked. Should be OK not to use sort
            arg_sort = torch.argsort(self.data.indices()[0])
            new_indices = self.data.indices()[:,arg_sort]
            new_values = self.data.values()[arg_sort]
        else:
            new_indices = self.data.indices()
            new_values = self.data.values()
        del self.data
        
        ele, counts = torch.unique(new_indices[0], return_counts=True)
        new_counts = torch.zeros([size[0]], dtype=torch.long)
        new_counts[ele] = counts
        new_counts = new_counts.tolist()
        new_indices = torch.split_with_sizes(new_indices[1], new_counts)
        new_values = torch.split_with_sizes(new_values, new_counts)
        
        bar = tqdm(total = size[0], ncols=0)
        def collate_fn(i, v):
            bar.update()
            return torch.sparse_coo_tensor(i[None,:], v, size=[size[1]])
        
        return [*map(collate_fn, new_indices, new_values)]




class tensor_retuen_type(dict):
    def __getattr__(self, item: str)->Tensor:
        try:
            return self[item]
        except KeyError:
            raise AttributeError
    def to(self, *args, **kwargs):
        return tensor_retuen_type(**{i:self[i].to(*args, **kwargs) for i in self})
    def __setattr__(self, name: str, value: torch.Any) -> None:
        self.__setitem__(name, value)
    def __setitem__(self, key, value:Tensor) -> None:
        if self.device is not None:
            value = value.to(self.device)
        return super().__setitem__(key, value)
    def __getstate__(self):
        return tensor_retuen_type(self)

    def __setstate__(self, state):
        super().__init__(state)
    
    def __len__(self,):
        return len([*self.values()][0])
    @property
    def device(self):
        if len(self.keys())>0:
            return next(iter(self.values())).device
        else:
            return None
        raise RuntimeError()
    

def Masking(x:Tensor, P:float, tokenizer, all_mask:Tensor=None)->tensor_retuen_type:
    x=x.clone()
    if all_mask is None:
        all_mask = torch.rand(x.shape, device=x.device) > P
    else:
        all_mask = all_mask.bool() * (torch.rand(x.shape, device=x.device) > P)
        
    all_mask[:,0], all_mask[:,-1] = 1, 1
    s = torch.rand(x.shape, device=x.device)
    mask_mask = (s<0.8) * ~all_mask
    rand_mask = ((s>0.8) * (s<0.9)) * ~all_mask
    
    x[mask_mask] = tokenizer.mask_token_id
    x[rand_mask] = torch.randint(999, tokenizer.vocab_size, size = x[rand_mask].shape, dtype=x.dtype, device=x.device)

    return tensor_retuen_type(input_ids = x, masks = all_mask.long(), attention_masks = generate_mask(x, tokenizer.pad_token_id))

if __name__=="__main__":
    t = tensor_retuen_type(a=torch.rand([4,16]), b=torch.rand([4,16]))
    t_=tensor_retuen_type()
    t = t.to('cuda')
    t.c = torch.rand([4,16])
    t['d'] = torch.rand([4,16])
    print(t)
    print(t.device)