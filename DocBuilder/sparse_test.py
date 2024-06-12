import sys
sys.path.append('..')
import torch
from utils import custom_sparse_mmT, top_k_sparse
import time
a = top_k_sparse(torch.randn([30522]), 128)
b = top_k_sparse(torch.randn([3000, 30522]), 128)

# a = torch.Tensor([1,0,1,1,0,1,1,0]).to_sparse()

# b = torch.exp2(torch.arange(8))
# b = torch.stack([b*2**i for i in range(4)])
# b = b.to_sparse()

print('The error of sparse opteration', torch.max(a.to_dense()@b.to_dense().T - custom_sparse_mmT(a, b)))

s = time.time()
for i in range(1000):
    a.to_dense()@b.to_dense().T
print(time.time()-s)
s = time.time()
for i in range(1000):
    custom_sparse_mmT(a,b)
print(time.time()-s)
    