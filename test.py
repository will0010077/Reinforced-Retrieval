import numpy as np
import torch.nn.functional as F
import torch
from torch import nn


@torch.no_grad()
def momentum_update(src:nn.Module, dst:nn.Module, factor=0.01):
    for s, d in zip(src.parameters(), dst.parameters()):
        d.data = (1-factor)*d.data + factor*s.data

a=torch.nn.Linear(5,10)
b=torch.nn.Linear(5,10)

print(a.weight, b.weight)
momentum_update(a,b,1)
print( b.weight)


