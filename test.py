import torch
a = torch.ones([1], dtype=torch.bfloat16)
for i in range(1000000):
    a = a*0.99
    print("\r",a, end="")