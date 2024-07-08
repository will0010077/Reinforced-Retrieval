# import numpy as np
import sys
sys.path.append('..')
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from DocBuilder.Retriever_k_means import inner
from DocBuilder.utils import sparse_retrieve_rep, tensor_retuen_type
from DatasetLoader.collate_func import collate
from LM.llama_reader import EncTunedLM
from DocBuilder.Retriever_k_means import doc_retriever
from metric.reward import BLEU_score, Bert_score
import random
import config
import torch.optim as optim
from transformers import BertModel, BertConfig, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.distributions import Categorical
from config import agent_size_config
from tqdm import tqdm
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
    
    
class LLMEnv_batch_version:
    def __init__(self, dataset, LM: EncTunedLM, ret: doc_retriever, action_space_size, history_len=6, batch_size=8):
        self.dataset = dataset  # List of tuples (x, y)
        self.action_space_size = action_space_size
        self.history_len = history_len
        self.LM = LM
        self.eos_id = self.LM.tokenizer.eos_token_id
        self.ret = ret
        self.current_index = 0
        self.collate = collate()
        self.batch_size = batch_size  # Set batch size as length of dataset
        self.x = [None] * self.batch_size
        self.y = [None] * self.batch_size
        self.d_t = [None] * self.batch_size
        self.basic_reward = [None] * self.batch_size
        self.log_prob = [None] * self.batch_size
        self.halulu = [None] * self.batch_size
        self.revise_reward = [None] * self.batch_size
        self.hat_y_t = [None] * self.batch_size
        self.response_cache = [None] * self.batch_size
        self.n = [None] * self.batch_size
        self.done = [None] * self.batch_size
        self.steps = [None] * self.batch_size
        self.last_action = [None] * self.batch_size
        self.last_proceed = [-1] * self.batch_size
        self.action_history = [None] * self.batch_size

    def reset(self, idx=None):
        if idx is None:
            for i in range(self.batch_size):
                self._reset_idx(i)
            return [self.get_state(i) for i in range(self.batch_size)]
        else:
            self._reset_idx(idx)
            return self.get_state(idx)

    def _reset_idx(self, idx):
        self.current_data = self.dataset[random.randrange(len(self.dataset))]
        self.x[idx], self.y[idx] = self.current_data
        self.y[idx] = self.y[idx].split(' ')
        chunk_size = 10
        self.y[idx] = [' '.join(self.y[idx][i:i + chunk_size]) for i in range(0, len(self.y[idx]), chunk_size)]
        self.d_t[idx], zt = self.ret.retrieve(self.x[idx], k=1, num_search=4)
        self.d_t[idx] = self.d_t[idx][0,0]
        self.basic_reward[idx] = Bert_score([self.get_basic_response(self.x[idx], " ".join(self.y[idx]), self.d_t[idx])[0]], [" ".join(self.y[idx])])[0]
        self.halulu[idx] = []
        self.revise_reward[idx] = []
        self.hat_y_t[idx] = None
        self.response_cache[idx] = [self.hat_y_t[idx]]
        self.n[idx] = -1  # Initialize n to -1
        self.done[idx] = False
        self.steps[idx] = 0
        self.last_action[idx] = -1
        self.last_proceed[idx] = -1
        self.action_history[idx] = []

    def get_state(self, idx) -> str:
        state = self.collate.state_templete(
            self.x[idx],
            self.cat_response(self.response_cache[idx][-self.history_len:]),
            self.d_t[idx][::2],
            self.action_history[idx][-self.history_len:]
        )
        return state

    def step(self, actions:Tensor):
        rewards = [0] * self.batch_size
        next_states = []

        retrieve_indices = []
        proceed_indices = []
        rewrite_indices = []
        action_verb=["retrieve","proceed","rewrite"]
        self.actions = actions.clone()
        for i, action in enumerate(actions):
            if not self.done[i]:
                self.action_history[i].append(action_verb[action])
                if action == 0:  # Retrieve Document
                    if self.last_action[i] != 0:
                        retrieve_indices.append(i)
                elif action == 1:  # Proceed Response
                    if self.n[i] + 1 < len(self.y[i]):
                        self.n[i] += 1
                        self.last_proceed[i] = self.steps[i]
                        proceed_indices.append(i)
                    else:
                        self.done[i]=True
                        self.actions[i]=-1
                elif action == 2:  # Rewrite Current Response
                    if self.n[i] > -1:
                        self.response_cache[i].pop()
                        rewrite_indices.append(i)

        # Process Retrieve Document actions
        
        if len(retrieve_indices)>0:
            q_t = [self.construct_query(i) for i in retrieve_indices]
            d_t, zt = self.ret.retrieve(q_t, k=1, num_search=4)
            d_t = d_t.squeeze(1)
            for idx,i in enumerate(retrieve_indices):
                self.d_t[i] = d_t[idx]

        # Process Proceed and Rewrite actions in a batch
        batch_indices = proceed_indices + rewrite_indices
        if batch_indices:
            responses, log_probs = self.get_next_response(batch_indices)
            for idx, i in enumerate(batch_indices):
                self.hat_y_t[i] = responses[idx]
                self.log_prob[i] = log_probs[idx]
                self.response_cache[i].append(self.hat_y_t[i])

        for i in range(self.batch_size):
            if self.steps[i]>3*len(self.y[i])+4:
                self.done[i]=True
        rewards = self.compute_reward()
        for i in range(self.batch_size):
            next_states.append(self.get_state(i))
            self.steps[i] += 1
        self.last_action = actions.clone()

        return next_states, rewards, self.done, {}

    def compute_reward(self, ):
        rewards = [0]*self.batch_size
        proceed_indices = []
        rewrite_indices = []
        for idx in range(self.batch_size):
            if self.actions[idx] == 1:
                # reward += Bert_score([self.cat_response(self.response_cache[idx][-1:])], [self.y[idx][self.n[idx]]])[0] / len(self.y[idx])
                proceed_indices.append(idx)
                self.halulu[idx].append(0.5 * self.log_prob[idx].exp().mean() / len(self.y[idx]))
            elif self.actions[idx] == 2:
                rewards[idx] -= 0.01
                if self.n[idx] > -1:
                    # reward += Bert_score([self.cat_response(self.response_cache[idx][-1:])], [self.y[idx][self.n[idx]]])[0] / len(self.y[idx])
                    rewrite_indices.append(idx)
                    self.halulu[idx].pop(-1)
                    self.halulu[idx].append(0.5 * self.log_prob[idx].exp().mean() / len(self.y[idx]))
                else:
                    rewards[idx] -= 0.05
                    
            if self.done[idx]:
                if self.n[idx] > -1:
                    rewards[idx] += 2 * (Bert_score([self.cat_response(self.response_cache[idx])], [" ".join(self.y[idx])])[0] - self.basic_reward[idx])
                    rewards[idx] += 0.1 * ((self.n[idx] + 1) / len(self.y[idx])) ** 2
                    rewards[idx] += 0.1 * sum(self.halulu[idx])
                    rewards[idx]=float(rewards[idx])
            
        batch_indices = proceed_indices + rewrite_indices
        if batch_indices:
            refs = [self.cat_response(self.response_cache[idx][-1:]) for idx in batch_indices]
            cands = [self.y[idx][self.n[idx]] for idx in batch_indices]
            batch_bert =  Bert_score(refs, cands)
            for i, idx in enumerate(batch_indices):
                rewards[idx] += batch_bert[i] / len(self.y[idx])
                rewards[idx] = float(rewards[idx])
                if rewards[idx]!=rewards[idx]:
                    print("NAN!!")
                    self.done[idx] = True
                    rewards[idx]=0
                rewards[idx] /=  self.steps[idx] - self.last_proceed[idx]+1
                for j in range(self.last_proceed[idx], self.steps[idx]):
                    self.revise_reward[idx][j] = float(rewards[idx])
        for idx in range(self.batch_size):
            self.revise_reward[idx].append(rewards[idx])
            
        return rewards

    def construct_query(self, idx):
        return self.x[idx] + self.cat_response(self.response_cache[idx][-self.history_len:])

    def cat_response(self, cache: list[Tensor]) -> str:
        if cache[0] is None:
            cache = cache[1:]
        if len(cache) == 0:
            return ""
        s = self.LM.tokenizer.decode(torch.cat(cache))
        return s

    def get_next_response(self, indices):
        messages = [self.collate.templete(self.x[i], self.cat_response(self.response_cache[i]))[0] for i in indices]
        answers = [self.y[i][self.n[i]] for i in indices]
        d_t = torch.stack([self.d_t[i] for i in indices])
        d_t = tensor_retuen_type(input_ids=d_t, attention_mask=torch.ones_like(d_t)).to(self.LM.device)

        responses, log_probs = self.LM.pseudo_generate(messages, answers, Doc_tokens=d_t, temperture=0.5, return_prob=True, decode=False)
        return responses, log_probs

    def get_basic_response(self, x, y, d_t):
        messages, answer = self.collate.templete(x, "")
        d_t = tensor_retuen_type(input_ids=d_t[None], attention_mask=torch.ones_like(d_t[None])).to(self.LM.device)
        response = self.LM.pseudo_generate(messages + " " + answer, y, Doc_tokens=d_t, temperture=0.5, return_prob=False, decode=True)
        return response



class LLMEnv:
    def __init__(self, dataset, LM:EncTunedLM, ret:doc_retriever, action_space_size, history_len = 6):
        self.dataset = dataset  # List of tuples (x, y)
        self.action_space_size = action_space_size
        self.history_len = history_len
        self.LM = LM
        self.eos_id = self.LM.tokenizer.eos_token_id
        self.ret = ret
        self.current_index = 0
        self.collate = collate()

    def reset(self):
        self.current_data = self.dataset[random.randrange(len(self.dataset))]
        self.current_index+=1
        self.x, self.y = self.current_data
        self.y:list[str] = self.y.split(' ')
        chunk_size = 10
        self.y = [' '.join(self.y[i:i+chunk_size]) for i in range(0,len(self.y), chunk_size)]
        self.d_t, zt = self.ret.retrieve(self.x, k=1, num_search=4)
        self.d_t = self.d_t.squeeze(1)
        self.basic_reward = Bert_score([self.get_basic_response(self.x, " ".join(self.y))[0]], [" ".join(self.y)])[0]
        self.halulu = []
        self.revise_reward = []
        self.hat_y_t = None
        self.response_cache = [self.hat_y_t]
        self.n = -1  # Initialize n to -1
        self.done = False
        self.steps = 0
        self.last_action=-1
        self.action_history = []
        return self.get_state()

    def get_state(self)->str:
        state = self.collate.state_templete(self.x, self.cat_response(self.response_cache[-self.history_len:]), self.d_t[...,::2], self.action_history)
        return state

    def step(self, action):
        reward = 0
        self.action = action
        if action == 0:  # Retrieve Document
            self.action_history.append("retrieve")
            if self.last_action!=0:
                q_t = self.construct_query()
                self.d_t, zt = self.ret.retrieve(q_t, k=1, num_search=4)
                self.d_t = self.d_t.squeeze(1)
                self.hat_y_t = self.hat_y_t  # Keep current response
        elif action == 1:  # Proceed Response
            self.action_history.append("proceed")
            if self.n+1 < len(self.y):
                self.n += 1
                self.hat_y_t, self.log_prob = self.get_next_response()
                self.response_cache.append(self.hat_y_t[0])
            else:
                self.done=True
        elif action == 2:  # Rewrite Current Response
            self.action_history.append("rewrite")
            if self.n>-1:
                self.response_cache.pop()
                self.hat_y_t, self.log_prob = self.get_next_response()
                self.response_cache.append(self.hat_y_t[0])
            else:
                reward+=-1
        elif action == 3:  # Output Answer
            self.done = True
        # if self.hat_y_t!=None and self.eos_id in self.hat_y_t[0]:
        #     self.done=True
        if len(self.action_history)>self.history_len:
            self.action_history.pop(0)
        if self.steps>3*len(self.y)+5:
            self.done=True
        reward = reward + self.compute_reward()
        next_state = self.get_state()
        self.steps += 1
        self.last_action=self.action
        return next_state, reward, self.done, {}

    def compute_reward(self):
        # Implement reward calculation logic
        reward=0
        if self.done:
            if self.n>-1:
                reward += 10*(Bert_score([self.cat_response(self.response_cache)], [" ".join(self.y)])[0] - self.basic_reward)
                reward += 0.1*((self.n+1)/len(self.y))**2
                reward += sum(self.halulu)
        # elif self.action == 0:
        #     if self.last_action!=0:
        #         reward += Bert_score([self.collate.datatokenizer.decode(self.d_t[0])], [" ".join(self.y)])[0]/len(self.y)
        if self.action==1:
            reward += Bert_score([self.cat_response(self.response_cache[-1:])], [self.y[self.n]])[0]/len(self.y)
            self.halulu.append(0.5*self.log_prob[0].exp().mean()/len(self.y))
        elif self.action==2:
            reward -= 0.01
            if self.n>-1:
                reward += Bert_score([self.cat_response(self.response_cache[-1:])], [self.y[self.n]])[0]/len(self.y)
                self.halulu.pop(-1)
                self.halulu.append(0.5*self.log_prob[0].exp().mean()/len(self.y))
                for i in reversed(range(self.steps)):
                    if self.revise_reward[i]>0:
                        self.revise_reward[i]=0
                        break
            else:
                reward+=-0.05
        reward = float(reward)
        self.revise_reward.append(reward)
        return reward
    
    def construct_query(self):
        self.x, self.d_t, self.response_cache
        # Implement query construction logic
        return self.x + self.cat_response(self.response_cache[-self.history_len:])

    def cat_response(self, cache:list[Tensor])->str:
        if cache[0]==None:
            cache = cache[1:]
        if len(cache)==0:
            return ""
        
        s = self.LM.tokenizer.decode(torch.cat(cache))
        return s
    def get_next_response(self,):
        # Implement response generation logic
        # messages, answer = self.collate.templete(self.x, ' '.join(self.response_cache))
        messages, answer = self.collate.templete(self.x, self.cat_response(self.response_cache))
        if self.d_t!=None:
            d_t = tensor_retuen_type(input_ids = self.d_t, attention_mask = torch.ones_like(self.d_t)).to(self.LM.device)
        # print("What is feeded:",messages+" "+answer, self.y[self.n])
        response, log_prob = self.LM.pseudo_generate(messages+" "+answer, self.y[self.n], Doc_tokens = d_t, temperture = 0.5, return_prob = True, decode = False)
        
        return response, log_prob
    def get_basic_response(self,x, y):
        # Implement response generation logic
        # messages, answer = self.collate.templete(self.x, ' '.join(self.response_cache))
        messages, answer = self.collate.templete(x, "")
        if self.d_t!=None:
            d_t = tensor_retuen_type(input_ids = self.d_t, attention_mask = torch.ones_like(self.d_t)).to(self.LM.device)
        # print("What is feeded:",messages+" "+answer, self.y[self.n])
        response = self.LM.pseudo_generate(messages+" "+answer, y, Doc_tokens = d_t, temperture = 0.5, return_prob = False, decode = True)
        
        return response

class LLMEnv_test(LLMEnv):
    
    
    def reset(self, x):
        self.x =x
        chunk_size = 10
        
        self.d_t, zt = self.ret.retrieve(self.x, k=1, num_search=4)
        self.d_t = self.d_t.squeeze(1)
        self.hat_y_t = None
        self.response_cache = [self.hat_y_t]
        self.done = False
        self.steps = 0
        self.last_action=-1
        self.action_history = []
        return self.get_state()
    
    def step(self, action):
        reward = 0
        self.action = action
        if action == 0:  # Retrieve Document
            self.action_history.append("retrieve")
            if self.last_action!=0:
                q_t = self.construct_query()
                self.d_t, zt = self.ret.retrieve(q_t, k=1, num_search=4)
                self.d_t = self.d_t.squeeze(1)
                self.hat_y_t = self.hat_y_t  # Keep current response
        elif action == 1:  # Proceed Response
            self.action_history.append("proceed")
            self.hat_y_t = self.get_next_response()
            self.response_cache.append(self.hat_y_t[0])
        elif action == 2:  # Rewrite Current Response
            self.action_history.append("rewrite")
            if len(self.response_cache)>1:
                self.response_cache.pop()
                self.hat_y_t = self.get_next_response()
                self.response_cache.append(self.hat_y_t[0])
        elif action == 3:  # Output Answer
            self.done = True
        if self.eos_id in self.hat_y_t[0]:
            self.done=True
        if len(self.action_history)>self.history_len:
            self.action_history.pop(0)
        self.steps += 1
        next_state = self.get_state()
        
        self.last_action=self.action
        return next_state, reward, self.done, {}
    def compute_reward(self):
        return 0
    def get_next_response(self,):
        # Implement response generation logic
        messages, answer = self.collate.templete(self.x, self.cat_response(self.response_cache))
        d_t = tensor_retuen_type(input_ids = self.d_t, attention_mask = torch.ones_like(self.d_t)).to(self.LM.device)
        response = self.LM.generate(messages+" "+answer, Doc_tokens = d_t, max_new_tokens=15, decode = False)
        
        return response
class BertAgentCritic(nn.Module):
    def __init__(self, model_config, action_space_size):
        super(BertAgentCritic, self).__init__()
        self.bert = RobertaModel.from_pretrained(config.roberta_dir, torch_dtype = torch.bfloat16).to(torch.bfloat16)
        self.tokenizer = RobertaTokenizer.from_pretrained(config.roberta_dir)
        self.action_head = nn.Linear(self.bert.config.hidden_size, action_space_size)
        self.value_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.action_space_size = action_space_size
    
    def forward(self, state = None, inputs = None):
        if inputs==None:
            inputs = self.tokenizer(state, return_tensors="pt", padding=True, truncation=True).to(self.bert.device)
        # inputs['input_ids'] shape: (batch_size, sequence_length)
        # inputs['attention_mask'] shape: (batch_size, sequence_length)
        outputs = self.bert(**inputs)
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        action_logits = self.action_head(cls_output).float()  # Shape: (batch_size, action_space_size)
        state_value = self.value_head(cls_output)[...,0].float()  # Shape: (batch_size,)
        return action_logits, state_value  # Shapes: (batch_size, action_space_size), (batch_size,)

class PPOTrainer:
    def __init__(self, model:BertAgentCritic, optimizer, gamma=0.99, clip_epsilon=0.2, lambd=0.95, update_epochs=4, batch_size=32, grad_step = 4):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lambd = lambd
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.grad_step = grad_step
        
        self.action_coef=1
        self.value_coef=2**-1
        self.entropy_coef=2**-6

    def ppo_loss(self, old_log_probs, dist:Categorical, batch_actions, advantages, returns, values):
        # old_log_probs shape: (batch_size,)
        # batch_action shape: (batch_size,)
        # advantages shape: (batch_size,)
        # returns shape: (batch_size,)
        # values shape: (batch_size,)
        new_log_probs = dist.log_prob(batch_actions)
        ratios = torch.exp(new_log_probs - old_log_probs)  # Shape: (batch_size,)
        surr1 = ratios * advantages  # Shape: (batch_size,)
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # Shape: (batch_size,)
        actor_loss = -torch.min(surr1, surr2).mean()  # Shape: scalar

        critic_loss = F.huber_loss(values, returns, "mean", 1.0)  # Shape: scalar
        entropy:Tensor = dist.entropy().mean() #scalar
        
        return actor_loss, critic_loss, - entropy  # Shape: scalar

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
    def f(self,batch):
        batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = zip(*batch)
        batch_token = self.model.tokenizer(batch_states, return_tensors = "pt", padding = True, truncation=True)
        batch_actions = torch.stack(batch_actions)
        batch_old_log_probs = torch.stack(batch_old_log_probs)
        batch_returns = torch.stack(batch_returns)
        batch_advantages = torch.stack(batch_advantages)
        return batch_token, batch_actions, batch_old_log_probs, batch_returns, batch_advantages
    def update(self, memory):
        old_states, old_actions, old_log_probs, rewards, dones, values = zip(*memory)
        returns = self.compute_gae(rewards, values, dones, next_value=0)

        old_states = old_states # Shape: (memory_size, state_size)
        old_actions = torch.tensor(old_actions, dtype=torch.long)  # Shape: (memory_size,)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)  # Shape: (memory_size,)
        returns = torch.tensor(returns, dtype=torch.float32)  # Shape: (memory_size,)
        values = torch.tensor(values, dtype=torch.float32)  # Shape: (memory_size,)
        advantages = returns - values  # Shape: (memory_size,)
        loader = DataLoader([*zip(old_states, old_actions, old_log_probs, returns, advantages)], self.batch_size, True, collate_fn=self.f, num_workers=1, pin_memory = True, persistent_workers=True, drop_last=True)
        step = 0
        bar = tqdm(total=self.update_epochs*len(loader), ncols=0)
        for _ in range(self.update_epochs):
            for batch_token, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
                step+=1
                batch_token = batch_token.to(self.model.bert.device)  # Shape: (batch_size, n)
                batch_actions = batch_actions.to(self.model.bert.device)  # Shape: (batch_size,)
                batch_old_log_probs = batch_old_log_probs.to(self.model.bert.device)  # Shape: (batch_size,)
                batch_returns = batch_returns.to(self.model.bert.device)  # Shape: (batch_size,)
                batch_advantages = batch_advantages.to(self.model.bert.device)  # Shape: (batch_size,)

                logits, state_values = self.model.forward(inputs = batch_token)  # logits shape: (batch_size, action_space_size), state_values shape: (batch_size, 1)
                dist = Categorical(logits=logits)  # Shape: (batch_size,)

                actor_loss, value_loss, entropy_loss = self.ppo_loss(batch_old_log_probs, dist, batch_actions, batch_advantages, batch_returns, state_values)  # Shape: scalar
                bar.set_postfix_str(f"ac: {actor_loss:.3f}, value: {value_loss:.3f}, entropy: {-entropy_loss:.3f}")
                bar.update()
                self.optimizer.zero_grad()
                loss:Tensor = self.action_coef*actor_loss+ self.value_coef*value_loss+ self.entropy_coef*entropy_loss
                loss.backward()
                if step%self.grad_step==0:
                    self.optimizer.step()


if __name__=='__main__':
    
    
    # Example usage
    env = LLMEnv()
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