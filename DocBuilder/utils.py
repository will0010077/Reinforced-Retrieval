import torch
from torch import Tensor
from tqdm import tqdm
import multiprocessing as mp
import time
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
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])

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
    output shape (M)
    '''
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
    inverted_index = torch.full([d], d, dtype=torch.long)
    inverted_index[indices_a]=torch.arange(k, dtype=torch.long)
    # print(inverted_index)#check OK
    
    # Map the filtered indices of b to new indices [0, k]
    # mapped_indices_b = torch.tensor([[idx[0], inverted_index[idx[1].item()]] for idx in filtered_indices_b.T]).T
    mapped_indices_b = filtered_indices_b
    mapped_indices_b[1] = inverted_index[mapped_indices_b[1]]
    # print(mapped_indices_b)#check OK
    
    # Create a dense matrix from mapped indices and values of b
    # b_dense = torch.sparse_coo_tensor(mapped_indices_b, filtered_values_b, (M, k)).to_dense()
    b_dense = torch.zeros([M,k], dtype=b.dtype)
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

def unbind_sparse(data:Tensor, use_sort=False):
    data = data.coalesce()
    size = data.shape
    if use_sort:
        arg_sort = torch.argsort(data.indices()[0])
        new_indices = data.indices()[:,arg_sort]
        new_values = data.values()[arg_sort]
    else:
        new_indices = data.indices()
        new_values = data.values()
        
    del data
    ele, counts = torch.unique(new_indices[0], return_counts=True)
    new_counts = torch.zeros([size[0]], dtype=torch.long)
    new_counts[ele] = counts
    new_indices = torch.split_with_sizes(new_indices[1], new_counts.tolist())
    new_values = torch.split_with_sizes(new_values, new_counts.tolist())
    
    bar = tqdm(total = size[0], ncols=0)
    def collate_fn(i, v):
        bar.update()
        return torch.sparse_coo_tensor(i[None,:], v, size=[size[1]])
    
    return [*map(collate_fn, new_indices, new_values)]
    