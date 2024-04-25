
import torch
from time import time
from DocBuilder.utils import split_list_to_batch, restore_batched_list, unbind_sparse
if __name__=='__main__':
    a = torch.randn([32,10]).to_sparse()
    print(a.sparse_dim(), a.dense_dim())
    a.sparse_resize_([64,10], a.sparse_dim(), a.dense_dim())
    print(a)
    exit()
    
    s=time()    
    a = torch.load('data/vecs_reduced_5000000.pt',mmap=False)
    print(time()-s)
    
    print(a[10000])
    
    s=time()
    # a = restore_batched_list(a)
    a = unbind_sparse(a)
    
    print(time()-s)
    print(a[10000])
    exit()
    s=time()
    # a = split_list_to_batch(a, bs=32)
    print(time()-s)
    s=time()
    torch.save(a, 'data/vecs_reduced_5000000.pt')
    print(time()-s)
    s=time()
    exit()
import sys
from transformers import AutoTokenizer
from LexMAE import lex_retriever
def top_k_sparse(x:torch.Tensor, k:int, vec_dim:int=-1):
    '''
    x: Tensor
    vec_dim: data dim, default -1
    out: sparsed x
    '''
    scale=len(x.shape)*2+1
    if scale>(x.shape[vec_dim]/k):
        print(f'Warning! Sparsed result larger than original Tensor. scale: {scale}, sparsity: {(x.shape[vec_dim]/k)}')
    assert k<=x.shape[vec_dim]# check k smaller than original size
    a, _=x.argsort(dim=vec_dim).split_with_sizes(split_sizes=[x.shape[vec_dim]-k, k], dim=vec_dim) #keep top k index
    x=x.scatter(dim=vec_dim, index=a, value=0)#other index full with zero
    x=x.to_sparse(layout=torch.sparse_coo)
    return x

lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to('cpu')
lex_MAE_retriver.model.load_state_dict(torch.load('app/save/LEX_MAE_retriever838.pt', map_location='cpu')['enc_model_state_dict'])
k=16

example='Breaking Bad is an American crime drama television series created and produced by Vince Gilligan for AMC. Set and filmed in Albuquerque, New Mexico, the series follows Walter White (Bryan Cranston), an underpaid, dispirited high-school chemistry teacher struggling with a recent diagnosis of stage-three lung cancer. White turns to a life of crime and partners with a former student, Jesse Pinkman (Aaron Paul), to produce and distribute methamphetamine to secure his family\'s financial future before he dies, while navigating the dangers of the criminal underworld. Breaking Bad premiered on AMC on January 20, 2008, and concluded on September 29, 2013, after five seasons consisting of 62 episodes.'
example='Breaking Bad premiered on AMC on January 20, 2008, and concluded on September 29, 2013, after five seasons consisting of 62 episodes.'
tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
z = lex_MAE_retriver.forward(tokens)
z = top_k_sparse(z, k)[0]
for i, v in sorted(zip(lex_MAE_retriver.tokenizer.decode(z.coalesce().indices()[0]).split(' '), z.coalesce().values()), key=lambda x:x[1], reverse=True):
    print(f'{i}:{v:.3f}, ',end='')
print('=============================================================')
example='How many seasons does Breaking Bad have?'
tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
z = lex_MAE_retriver.forward(tokens)
z = top_k_sparse(z, k)[0]
for i, v in sorted(zip(lex_MAE_retriver.tokenizer.decode(z.coalesce().indices()[0]).split(' '), z.coalesce().values()), key=lambda x:x[1], reverse=True):
    print(f'{i}:{v:.3f}, ',end='')

exit(0)


# torch.save(a,'tensor.pt')
# torch.save(b,'sparse_tensor.pt')