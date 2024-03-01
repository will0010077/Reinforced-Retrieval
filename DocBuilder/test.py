from Retriever import cluster_builder, cos_sim
import torch
from tqdm import tqdm
import time
import os
print(os.getcwd())
print(os.path.dirname(__file__))

class test:
    def __init__(self, x):
        while True:
            _,x = x.split([10,x.shape[0]-10])
            print('\r',x.shape)
            time.sleep(0.1)
            
cluster = cluster_builder(num_search=4, k=int(1*10**3),bs=10**5, lr=0.1)
dim=768
data=torch.cat([torch.randn([10**5,dim], device='cuda').cpu() for _ in range(10)])
# test(data)
'''-------------------------------------------'''
cluster_ids_x, centers=cluster.train(data, epoch=10, bs = 10**5, tol=0.1, lr=0.2)
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
    
    