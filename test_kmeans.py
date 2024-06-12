from DocBuilder.Retriever_k_means import cluster_builder
from DocBuilder.utils import top_k_sparse
import torch
from tqdm import tqdm
import time
import os

cluster = cluster_builder(k=int(1*10**3))
dim=768
data=[]
for _ in tqdm(range(10**5)):
    data.extend(top_k_sparse(torch.randn([10**1,dim], device='cuda'), 128).cpu())
print(len(data))
# test(data)
'''-------------------------------------------'''
cluster_ids_x, centers=cluster.train(data, epoch=10, bs = 10**4, tol=0.1, lr=0.2)
cluster.build()
# cluster.save(t=87)
# cluster.load(t=87)
'''-------------------------------------------'''

bs=10
k=50
print(f'searching with batch size:{bs} and k:{k}...')
for i in tqdm(range(10000)):
    idx, emb = cluster.search(torch.randn([bs,dim], device='cuda'), k)
    # print(idx)

