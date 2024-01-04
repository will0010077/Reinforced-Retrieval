import torch
from tqdm import tqdm
'''

Given document D, dataset (q_i,y_i)
Compute latents Z={z_1,..., z_N\} = Emb(d_1)
While not converge
----Given a query $q$, hidden state z_q = Emb(q)
----for t in 1 to k:
--------Kt=Vt=Concat(z_q,z1,...,z(t-1))
--------Perturbed embedding zt Transformer(z_q, Kt, Vt)
--------top n subset D_q=top n(D,z_q^t)
--------Retrieve document (dt,zt) from policy.
----end for
----get y hat = LM(q, d1,...,dk)
----loss = CE(y hat, y)
----update policy and value loss
end while

'''
# init retriever
retriever:torch.Module
# pre compute embbeding


# init module
encoder:torch.Module
LM:torch.Module
perturb:torch.Module
optim:torch.optim.Optimizer

# init agent
policy:torch.Module


max_epoch = 10
num_retrieve=5
train_bar=tqdm()
for epoch in range(max_epoch):
    for q,y in train_bar:
        zq = retriever.emb(q)
        K=q, V=q
        doc_set = []
        for k in num_retrieve:
            
            qt = perturb(q, K, V)
            
            # zt, dt = retrieve(qt)
            doc_set.append()
            # K, V = concat K,V,q
        
        
        #forward and loss
        enc = encoder(doc_set)
        y, loss = LM(q,enc)
        
        reward = -loss# temperaly
        
        # grad update
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        
        # replay buffer
        replay.append(transition(q, doc_set, prob, reward))
        
        if RL_update:
            policy.update()
            replay.clear()
        
    





