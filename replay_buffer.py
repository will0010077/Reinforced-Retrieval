# import numpy as np
import torch


class transition:
    def __init__(self, inputs, outputs, retrieved, rewards):
        self.inputs, self.outputs, self.retrieved, self.rewards = inputs, outputs, retrieved, rewards

    def __str__(self) -> str:
        return f'inputs:{self.inputs.shape}, outputs:{self.outputs.shape}, retrieved:{self.retrieved.shape}, rewards:{self.rewards.shape}'


class doc_buffer:
    def __init__(self,):
        self.clear()
    
    def append(self, t):
        '''
        (in emb(d), policy output (k,d), retrieved embedding (k,d) or id (k), reward r, )\\
        policy output for important sampling\\
        reward function r(q, predict, y, z1,zt, )\\
        '''
        self.buffer.append(t)
        
        pass
    
    def stack(self, i):
        return torch.stack([getattr(x, i) for x in self.buffer])
    def sample(self,):
        
        return transition(self.stack('inputs'), 
                          self.stack('outputs'), 
                          self.stack('retrieved'), 
                          self.stack('rewards'))
        
    
    def clear(self,):
        self.buffer = []


if __name__=='__main__':
    
    B = doc_buffer()
    
    for i in range(10000):
        B.append(transition(torch.rand([64]), torch.rand([5,64]), torch.rand([5,64]), torch.ones(1)))
    B.clear()
    for i in range(10000):
        B.append(transition(torch.rand([64]), torch.rand([5,64]), torch.rand([5,64]), torch.ones(1)))
    
    print(B.sample())
    