
import torch.nn.functional as F
import torch
from torch import nn

dim=10
Q,K,V=torch.eye(10),torch.eye(10),torch.eye(10)
Q+=torch.randn([10,10])*0.001
K+=torch.randn([10,10])*0.001
V+=torch.randn([10,10])*0.001

I=torch.eye(10)+torch.randn([10,10])*0.001
O=torch.eye(10)+torch.randn([10,10])*0.001
x=torch.randn([3,10])

print(x)
x=x@I

q=x@Q
k=x@K
v=x@V

w=torch.softmax(q@k.T/dim**0.5, dim=-2)[:,:,None]
x=torch.sum(w*v[None,...], dim=0)
x=x@O
print(x)

