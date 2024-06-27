# import numpy as np
import sys
sys.path.append('..')
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from DocBuilder.Retriever_k_means import inner
from DocBuilder.utils import sparse_retrieve_rep, tensor_retuen_type
from DatasetLoader.collate_func import collate
from LM.llama_reader import EncTunedLM
from DocBuilder.Retriever_k_means import doc_retriever
from metric.reward import BLEU_score, Bert_score
import random

import torch.optim as optim
from transformers import BertModel, BertConfig, BertTokenizer
from torch.distributions import Categorical
from config import agent_size_config

class transition(tensor_retuen_type):
    def __init__(self, *args, **kwargs):
        '''
        inputs
        preds
        ret
        neg
        rewards
        '''
        super().__init__(*args, **kwargs)
        
    
    def __getstate__(self,):
        return self
    def __setstate__(self, state):
        self.update(state)
        
    def __str__(self) -> str:
        return f'inputs:{self.inputs.shape}, preds:{self.preds.shape}, ret:{self.ret.shape}, neg:{self.neg.shape}, rewards:{self.rewards.shape}'


class doc_buffer:
    def __init__(self, max_len=2**14):
        self.clear()
        self.max_len=max_len
    
    def append(self, t):
        '''
        adding a transition to buffer
        '''
        self.buffer.append(t)
        if len(self)>self.max_len:
            self.buffer.pop(0)
    
    def stack(self, name, s:Tensor = None):
        if s is not None:
            return torch.stack([self.buffer[i][name] for i in s])
        return torch.stack([getattr(x, name) for x in self.buffer])
    
    def sample(self, bs, shuffle = False):
        if shuffle:
            index = torch.randperm(len(self))
        else:
            index = torch.arange(len(self))
        
        for i in range(0, len(self), bs):
            yield transition(**{k: self.stack(k, index[i:i+bs]) for k in self.buffer[0]})

        
    def __len__(self,):
        return len(self.buffer)
    def clear(self,):
        self.buffer = []


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000, scale=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe*scale
        self.register_buffer('pe', pe)
        self.pe:Tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x+ self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class perturb_model(nn.Module):
    
    def __init__(self, in_dim=768, dim=768, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim, dropout=dropout,batch_first=True)
        self.model=torch.nn.TransformerEncoder(self.layer, num_layers)
        # self.model = torch.nn.ModuleList([torch.nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.pos_encoder = PositionalEncoding(dim, dropout=dropout, max_len=16)
        self.dim=dim
        self.in_dim=in_dim
        
        self.scale1=torch.nn.Linear(in_dim, dim, bias=True)
        self.scale2=torch.nn.Linear(dim, in_dim, bias=True)
        torch.nn.init.zeros_(self.scale2.weight.data)
        torch.nn.init.zeros_(self.scale2.bias.data)
        self.value = torch.nn.Linear(dim, 1)
                    
    def forward(self, x:torch.Tensor, mask=None)->Tensor:
        '''
        x: (B,n,d)
        mask: (n,n)
        out: shape of x
        '''
        x = self.scale1(x)
        x = self.pos_encoder(x)
        
        if mask is None:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device)
        x=self.model.forward(x, mask)# + x #  (torch.nn.functional.sigmoid(self.lam))
            
        out = self.scale2(x)
        value = self.value(x)
        return out, value
    
    
class Transformer_Agent(nn.Module):
    def __init__(self, in_dim, dim=768, **kwargs):
        super().__init__()
        self.model = perturb_model(in_dim, dim, **kwargs)
    def forward(self, x):
        '''
        forward output y with nromalize and value
        '''
        y, v=self.model.forward(x)
        y = F.relu_(y+x)
        y = F.normalize(y, dim=-1)
        return y, v
    
    @torch.no_grad()
    def next(self, x:torch.Tensor):
        '''
        x: (B,n,d)
        output: (B,d)
        '''
        x, v = self.forward(x)
        
        return x[:,-1,:]
    
    def get_loss(self, t:transition)->Tensor:
        '''
        Reinforce algorithm
        return : loss (B,k)
        '''
        t.neg#(([32, 5, 16, 30522]))
        outputs, value = self.forward(t.inputs)#(32,5,30522)
        temperture = 1
        # get neg
        neg = (outputs[:,:,None,:] @ t.neg.permute([0,1,3,2])).squeeze(-2)/temperture#(32,5,16)
        pos = (outputs[:,:,None,:] @ t.ret[...,None])[:,:,0,0]/temperture#(32,5)
        # get maximum number to prevent overflow
        M = torch.max(neg, dim=-1, keepdim=True).values#(32,5,1)
        # log_softmax function
        log_pi = pos - M[:,:,0] - (neg-M).exp().sum(-1).log()
        
        adv = t.rewards[:,None] - value[...,0]
        v_loss = adv**2
        pi_loss = - adv.detach() * log_pi
        # regularization to original input query
        reg_loss = ((outputs - t.inputs).norm(dim=-1))
        # FLOPS loss
        flops_loss = torch.abs(outputs).sum(-1)**2
        return pi_loss, v_loss, reg_loss, flops_loss   

class CustomEnv:
    def __init__(self, dataset, LM:EncTunedLM, ret:doc_retriever, action_space_size):
        self.dataset = dataset  # List of tuples (x, y)
        self.action_space_size = action_space_size
        self.LM = LM
        self.ret = ret
        self.current_index = 0
        self.collate = collate()
        self.reset()

    def reset(self):
        self.current_data = self.dataset[self.current_index % len(self.dataset)]
        self.current_index+=1
        self.x, self.y = self.current_data
        self.y:list[str] = self.y.split(' ')
        chunk_size = 10
        self.y = [' '.join(self.y[i:i+chunk_size]) for i in range(0,len(self.y), chunk_size)]
        
        self.d_t, zt = self.ret.retrieve(self.x, k=1, num_search=4)
        self.d_t = self.d_t.squeeze(1)
        self.hat_y_t = ""
        self.response_cache = [self.hat_y_t]
        self.n = -1  # Initialize n to -1
        self.done = False
        self.steps = 0
        self.last_action=-1
        return self.get_state()

    def get_state(self)->str:
        state = self.collate.state_templete(self.x, " ".join(self.response_cache), self.d_t)
        return state

    def step(self, action):
        reward = 0
        self.action = action
        if action == 0:  # Retrieve Document
            q_t = self.construct_query()
            self.d_t, zt = self.ret.retrieve(q_t, k=1, num_search=4)
            self.d_t = self.d_t.squeeze(1)
            self.hat_y_t = self.hat_y_t  # Keep current response
        elif action == 1:  # Get Next Response
            if self.n+1 < len(self.y):
                self.n += 1
                self.hat_y_t = self.get_next_response()
                self.response_cache.append(self.hat_y_t[0])
                if self.n+1==len(self.y):
                    self.done = True
            elif self.n+1 == len(self.y):
                reward += -1 # Panalty to too long response
                self.done=True
        elif action == 2:  # Rewrite Current Response
            if self.n>-1:
                self.response_cache.pop()
                self.hat_y_t = self.get_next_response()
                self.response_cache.append(self.hat_y_t[0])
            else:
                reward+=-1
        elif action == 3:  # Output Answer
            self.done = True

        self.steps += 1
        if self.steps>2*len(self.y):
            self.done=True
        reward = reward + self.compute_reward()
        next_state = self.get_state()
        
        self.last_action=self.action
        return next_state, reward, self.done, {}

    def construct_query(self):
        self.x, self.d_t, self.response_cache
        # Implement query construction logic
        return self.x + ' '.join(self.response_cache)

    def retrieve_document(self, query):
        # Implement document retrieval logic
        return f"Document for {query}"

    def get_next_response(self,):
        # Implement response generation logic
        self.x, self.response_cache
        messages, answer = self.collate.templete(self.x, ' '.join(self.response_cache))
        d_t = tensor_retuen_type(input_ids = self.d_t, attention_mask = torch.ones_like(self.d_t)).to(self.LM.device)
        response = self.LM.pseudo_generate(messages+" "+answer, self.y[self.n], Doc_tokens = d_t)
        
        return response

    def compute_reward(self):
        # Implement reward calculation logic
        reward=0
        if self.action == 0:
            if self.last_action!=self.action:
                reward += Bert_score([self.collate.datatokenizer.decode(self.d_t[0])], [self.y[self.n]])[0]/len(self.y)
            else:
                reward += 0
        elif self.action==1:
            reward += Bert_score([self.response_cache[-1]], [self.y[self.n]])[0]/len(self.y)
        elif self.action==2:
            if self.n>-1:
                reward += Bert_score([self.response_cache[-1]], [self.y[self.n]])[0]/len(self.y)
        elif self.action==3 or self.done:
            reward += (reward/len(self.y))**4
        
        return float(reward)

class BertAgentCritic(nn.Module):
    def __init__(self, model_config, action_space_size):
        super(BertAgentCritic, self).__init__()
        self.bert = BertModel(BertConfig(**model_config))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.action_head = nn.Linear(self.bert.config.hidden_size, action_space_size)
        self.value_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.action_space_size = action_space_size
    
    def forward(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.bert.device)
        # inputs['input_ids'] shape: (batch_size, sequence_length)
        # inputs['attention_mask'] shape: (batch_size, sequence_length)
        outputs = self.bert(**inputs)
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        action_logits = self.action_head(cls_output)  # Shape: (batch_size, action_space_size)
        state_value = self.value_head(cls_output)  # Shape: (batch_size, 1)
        return action_logits, state_value  # Shapes: (batch_size, action_space_size), (batch_size, 1)

class PPOTrainer:
    def __init__(self, model, optimizer, gamma=0.99, clip_epsilon=0.2, lambd=0.95, update_epochs=4, batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lambd = lambd
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def ppo_loss(self, old_log_probs, log_probs, advantages, returns, values):
        # old_log_probs shape: (batch_size,)
        # log_probs shape: (batch_size,)
        # advantages shape: (batch_size,)
        # returns shape: (batch_size,)
        # values shape: (batch_size,)

        ratios = torch.exp(log_probs - old_log_probs)  # Shape: (batch_size,)
        surr1 = ratios * advantages  # Shape: (batch_size,)
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # Shape: (batch_size,)
        actor_loss = -torch.min(surr1, surr2).mean()  # Shape: scalar

        critic_loss = (returns - values).pow(2).mean()  # Shape: scalar

        entropy_loss = -log_probs.mean()  # Shape: scalar
        return actor_loss + 2. * critic_loss - 0.1 * entropy_loss  # Shape: scalar

    def compute_gae(self, rewards, values, dones, next_value):
        # rewards shape: (sequence_length,)
        # values shape: (sequence_length,)
        # dones shape: (sequence_length,)
        # next_value shape: scalar

        values = values + (next_value,)  # Shape: (sequence_length + 1,)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]  # Shape: scalar
            gae = delta + self.gamma * self.lambd * (1 - dones[step]) * gae  # Shape: scalar
            returns.insert(0, gae + values[step])  # Shape: scalar
        return returns  # Shape: (sequence_length,)

    def update(self, memory):
        old_states, old_actions, old_log_probs, rewards, dones, values = zip(*memory)
        returns = self.compute_gae(rewards, values, dones, next_value=0)

        old_states = old_states # Shape: (memory_size, state_size)
        old_actions = torch.tensor(old_actions, dtype=torch.float32)  # Shape: (memory_size,)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)  # Shape: (memory_size,)
        returns = torch.tensor(returns, dtype=torch.float32)  # Shape: (memory_size,)
        values = torch.tensor(values, dtype=torch.float32)  # Shape: (memory_size,)

        advantages = returns - values  # Shape: (memory_size - 1,)

        for _ in range(self.update_epochs):
            for idx in range(0, len(old_states), self.batch_size):
                batch_states = old_states[idx:idx+self.batch_size]  # Shape: (batch_size, state_size)
                batch_actions = old_actions[idx:idx+self.batch_size].to(self.model.bert.device)  # Shape: (batch_size,)
                batch_old_log_probs = old_log_probs[idx:idx+self.batch_size].to(self.model.bert.device)  # Shape: (batch_size,)
                batch_returns = returns[idx:idx+self.batch_size].to(self.model.bert.device)  # Shape: (batch_size,)
                batch_advantages = advantages[idx:idx+self.batch_size].to(self.model.bert.device)  # Shape: (batch_size,)

                logits, state_values = self.model(batch_states)  # logits shape: (batch_size, action_space_size), state_values shape: (batch_size, 1)
                log_probs = Categorical(logits=logits).log_prob(batch_actions)  # Shape: (batch_size,)

                loss = self.ppo_loss(batch_old_log_probs, log_probs, batch_advantages, batch_returns, state_values)  # Shape: scalar

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


if __name__=='__main__':
    
    
    # Example usage
    env = CustomEnv()
    model = BertAgentCritic(agent_size_config, env.action_space_size)
    trainer = PPOTrainer(model)

    # Training loop
    for episode in range(1000):
        state = env.reset()  # Shape: string
        done = False
        memory = []

        while not done:
            action_logits, state_value = model([state])  # action_logits shape: (1, action_space_size), state_value shape: (1, 1)
            action_prob = torch.softmax(action_logits, dim=-1)  # Shape: (1, action_space_size)
            dist = Categorical(action_prob)
            action = dist.sample()  # Shape: (1,)

            next_state, reward, done, _ = env.step(action.item())  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            memory.append((state, action, dist.log_prob(action), reward, done, state_value))  # Shapes: (string, (1,), (1, action_space_size), scalar, scalar (boolean), (1, 1))

            state = next_state

        trainer.update(memory)