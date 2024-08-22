# import numpy as np
import sys
sys.path.append('..')
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from DocBuilder.Retriever_k_means import inner
from DocBuilder.Retriever_k_means import doc_retriever
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import sparse_retrieve_rep, tensor_retuen_type
from DatasetLoader.collate_func import collate
from LM.llama_reader import EncTunedLM
from metric.reward import metric
import random
import config
import torch.optim as optim
from transformers import BertModel, BertConfig, BertTokenizer, RobertaModel, RobertaTokenizer, RobertaForMaskedLM
from torch.distributions import Categorical
from config import agent_size_config
from tqdm import tqdm
from itertools import chain

    
def generate_segments(text:str, window_size, step)-> list[str]:
    if isinstance(text, list):
        text=" ".join(text)
    text = text.split()
    segment_list=[]

    for i in range(0, max(len(text)-window_size,1), step):
        segment_data = text[max(0, min(i, len(text)-window_size)):i+window_size]
        # print(segment_data.shape)
        segment_list.append(" ".join(segment_data))
    return  segment_list
class LLMEnv_batch_version(nn.Module):
    def __init__(self, dataset, LM: EncTunedLM, ret: lex_retriever, action_space_size, collate, history_len=4, batch_size=8, shuffle = True, step_size=15, eos_id = None):
        super().__init__()
        self.metric = metric()
        self.dataset = dataset  # List of tuples (x, y)
        self.action_space_size = action_space_size
        self.history_len = history_len
        self.LM = LM
        if eos_id is None:
            self.eos_id = self.LM.tokenizer.eos_token_id
        self.ret = ret
        self.current_index = 0
        self.collate = collate
        self.batch_size = batch_size  # Set batch size as length of dataset
        self.action_verb=[" retrieve", " proceed", " rewrite"]
        self.step_size=step_size
        
        
        self.x = [None] * self.batch_size
        self.y = [None] * self.batch_size
        self.ground_truth = [None] * self.batch_size
        self.document = [None] * self.batch_size
        self.input_ids = [None] * self.batch_size
        self.attention_mask = [None] * self.batch_size
        self.embedding = [None] * self.batch_size
        
        self.d_t = [None] * self.batch_size
        self.basic_reward = [None] * self.batch_size
        self.probs = [None] * self.batch_size
        self.probs_halulu = [None] * self.batch_size
        self.reward = [None] * self.batch_size
        self.previous_generation_quality = [None] * self.batch_size
        
        self.hat_y_t = [None] * self.batch_size
        self.response_cache = [None] * self.batch_size
        self.n = [None] * self.batch_size
        self.done = [None] * self.batch_size
        self.steps = [None] * self.batch_size
        self.last_action = [None] * self.batch_size
        self.action_history = [None] * self.batch_size
        self.last_proceed = [-1] * self.batch_size
        self.shuffle = shuffle

    def reset(self, idx=None):
        if idx is None:
            for i in range(self.batch_size):
                self._reset_idx(i)
            return [self.get_state(i) for i in range(self.batch_size)]
        else:
            self._reset_idx(idx)
            return self.get_state(idx)
    @torch.no_grad()
    def _build_embedding(self, idx):
        self.document[idx] = generate_segments(self.document[idx],config.data_config.windows, config.data_config.step)[:512]
        self.input_ids[idx] = []
        self.attention_mask[idx] = []
        self.embedding[idx] = []
        for i in range(0,len(self.document[idx]), 32):
            tokens = self.collate.datatokenizer(self.document[idx][i:i+32], padding = True, truncation=True, max_length=256, return_tensors="pt", add_special_tokens=False).to(self.ret.device)
            self.input_ids[idx].extend(tokens.input_ids)
            self.attention_mask[idx].extend(tokens.attention_mask)
            self.embedding[idx].extend(self.ret.forward(tokens)) #(N,d)
        self.embedding[idx] = torch.stack(self.embedding[idx])
    @torch.no_grad()
    def retrieve(self, ids:list, x:list[str]):
        query = self.ret.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.ret.device)
        query = self.ret.forward(query)#(b,d)
        retrieved = []
        for idx, q in zip(ids, query):
            topk = torch.argmax(q[None] @ self.embedding[idx].T, dim=-1)[0]#(1,1)->()
            retrieved.append(self.input_ids[idx][topk][:sum(self.attention_mask[idx][topk])])
        return retrieved
        
    def _reset_idx(self, idx):
        if self.shuffle:
            self.current_data = self.dataset[random.randrange(len(self.dataset))]
        else:
            self.current_data = self.dataset[self.current_index%len(self.dataset)]
        self.current_index+=1
        self.x[idx], self.y[idx], self.document[idx] = self.current_data
        self.ground_truth[idx] = self.y[idx]
        if isinstance(self.y[idx],list):
            self.y[idx]=self.y[idx][0]
        self._build_embedding(idx)
        self.y[idx] = self.y[idx].split(' ')
        chunk_size = 10
        if len(self.y[idx]) > 10*chunk_size:
            chunk_size = len(self.y[idx])//6+1
            
        self.y[idx] = [' '.join(self.y[idx][i:i + chunk_size]) for i in range(0, len(self.y[idx]), chunk_size)]
        self.d_t[idx] = self.retrieve([idx], self.x[idx])[0]
        self.basic_reward[idx] = self.metric.Bert_score([self.get_basic_response(self.x[idx], " ".join(self.y[idx]), self.d_t[idx])[0]], [" ".join(self.y[idx])])[0]
        self.reward[idx] = []
        self.previous_generation_quality[idx] = 0
        
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
            [self.action_verb[i] for i in self.action_history[idx][-self.history_len:]],
            self.d_t[idx]
        )
        return state

    def step(self, actions:Tensor):
        rewards = [0] * self.batch_size
        next_states = []

        retrieve_indices = []
        proceed_indices = []
        rewrite_indices = []
        self.actions = actions.clone()
        for i, action in enumerate(actions):
            if not self.done[i]:
                if action == 0:  # Retrieve Document
                    if self.last_action[i] != 0:
                        retrieve_indices.append(i)
                elif action == 1:  # Proceed Response
                    if self.n[i] + 1 < len(self.y[i]):
                        # if self.response_cache[i][-1] is not None and self.eos_id in self.response_cache[i][-1]:
                        #     self.done[i] = True
                        #     pass
                        # else:
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
                self.action_history[i].append(action)

        # Process Retrieve Document actions
        
        if len(retrieve_indices)>0:
            q_t = [self.construct_query(i) for i in retrieve_indices]
            d_t = self.retrieve(retrieve_indices, q_t)
            for idx, i in enumerate(retrieve_indices):
                self.d_t[i] = d_t[idx]

        # Process Proceed and Rewrite actions in a batch
        batch_indices = proceed_indices + rewrite_indices
        if batch_indices:
            responses, token_prob_input, token_prob_resampled = self.get_next_response(batch_indices)
            for idx, i in enumerate(batch_indices):
                self.hat_y_t[i] = responses[idx]
                self.probs[i] = token_prob_input[idx]
                self.probs_halulu[i] = token_prob_resampled[idx]
                self.response_cache[i].append(self.hat_y_t[i])

        for i in range(self.batch_size):
            if self.steps[i]>3*len(self.y[i])+3:
                self.done[i]=True
        rewards = self.compute_reward()
        for i in range(self.batch_size):
            self.reward[i].append(rewards[i])
        for i in range(self.batch_size):
            next_states.append(self.get_state(i))
            self.steps[i] += 1
        self.last_action = self.actions.clone()

        return next_states, rewards, self.done, {}


    def generation_quality(self, idx):
        return 1.0*self.probs[idx].mean().exp() + 0.1*self.probs_halulu[idx].mean().exp()
    
    def retrieval_quality(self, idx):
        return self.metric.BLEU_1_score([self.ret.tokenizer.decode(self.d_t[idx], skip_special_tokens=True)], [" ".join(self.y[idx])])[0]
    def compute_reward(self, ):
        rewards = [0]*self.batch_size
        proceed_indices = []
        rewrite_indices = []
        cands=[]
        refs=[]
        bert_idx=[]
        for idx in range(self.batch_size):
            if self.done[idx]:
                if self.n[idx] > -1:
                    cands.append(self.cat_response(self.response_cache[idx]))
                    refs.append(" ".join(self.y[idx]))
                    bert_idx.append(idx)
                    rewards[idx] += - self.basic_reward[idx]
                    rewards[idx] = float(rewards[idx])
                    
                        
            elif self.actions[idx] == 0:
                # retrieval score
                if self.last_action[idx] != 0 and self.action_history[idx].count(0)<len(self.y[idx]):
                    rewards[idx] += 0.1*self.retrieval_quality(idx) / len(self.y[idx])
                if self.last_action[idx]==0:
                    rewards[idx] -= 0.005
            elif self.actions[idx] == 1:
                temp = self.generation_quality(idx)
                rewards[idx] += 0.05*temp/len(self.y[idx])
                rewards[idx] += 0.05 * 1/ len(self.y[idx])
                self.previous_generation_quality[idx] = temp
                
            elif self.actions[idx] == 2:
                if self.n[idx] > -1:
                    temp = self.generation_quality(idx)
                    rewards[idx] += 0.05*max(temp - self.previous_generation_quality[idx], 0) / len(self.y[idx])
                    self.previous_generation_quality[idx] = temp
                else:
                    rewards[idx] -= 0.01
        
        if cands:
            bert = self.metric.Bert_score(cands, refs)
            for i, idx in enumerate(bert_idx):
                rewards[idx] += bert[i]
            
        for idx in range(self.batch_size):
            rewards[idx] = float(rewards[idx])
            if rewards[idx]!=rewards[idx]:
                print("reward NAN!!")
                self.done[idx] = True
                rewards[idx]=0
            
            
        return rewards

    def construct_query(self, idx):
        return self.x[idx] + self.cat_response(self.response_cache[idx][-self.history_len:])

    def cat_response(self, cache: list[Tensor], skip_special_tokens=False) -> str:
        if cache[0] is None:
            cache = cache[1:]
        if len(cache) == 0:
            return ""
        s = self.LM.tokenizer.decode(torch.cat(cache), skip_special_tokens=skip_special_tokens)
        return s

    def get_next_response(self, indices):
        # response = [self.cat_response(self.response_cache[i]) for i in indices]
        response = [" ".join(self.y[i][:self.n[i]]) for i in indices]
        messages = [" ".join(self.collate.templete(self.x[i], response[idx])) for idx, i in enumerate(indices)]
        answers = [self.y[i][self.n[i]] for i in indices]
        # d_t = torch.stack([self.d_t[i] for i in indices])#need to padding
        d_t = self.ret.tokenizer.batch_decode([self.d_t[i] for i in indices], skip_special_tokens=True)
        doc_token = self.ret.tokenizer(d_t, return_tensors="pt", padding=True).to(self.LM.device)
        # d_t = tensor_retuen_type(input_ids=d_t, attention_mask=torch.ones_like(d_t)).to(self.LM.device)
        messages = [messages[j].replace(" </knowledge>", d_t[j]+" </knowledge>") for j in range(len(messages))]

        responses, token_prob_input, token_prob_resampled = self.LM.pseudo_generate(messages, answers, Doc_tokens=doc_token, temperture=0.5, return_prob=True, decode=False)
        return responses, token_prob_input, token_prob_resampled

    def get_basic_response(self, x, y, d_t):
        messages, answer = self.collate.templete(x, "")
        d_t = tensor_retuen_type(input_ids=d_t[None], attention_mask=torch.ones_like(d_t[None])).to(self.LM.device)
        response = self.LM.pseudo_generate(messages, y, Doc_tokens=d_t, temperture=0.01, return_prob=False, decode=True)
        return response



class LLMEnv_test(LLMEnv_batch_version):
    
    def step(self, actions:Tensor):
        rewards = [0] * self.batch_size
        next_states = []

        retrieve_indices = []
        proceed_indices = []
        rewrite_indices = []
        self.actions = actions.clone()
        for i, action in enumerate(actions):
            if not self.done[i]:
                if  len(self.y[i])-self.steps[i] > self.n[i]:
                    action = 1
                    self.actions[i]=1
                if action == 0:  # Retrieve Document
                    if self.last_action[i] != 0:
                        retrieve_indices.append(i)
                    else:
                        action=1
                        self.actions[i]=1
                if action == 1:  # Proceed Response
                    if self.hat_y_t[i] is not None and self.eos_id in self.hat_y_t[i]:
                        self.done[i] = True
                        continue
                    self.n[i] += 1
                    proceed_indices.append(i)
                elif action == 2:  # Rewrite Current Response
                    if self.n[i] > -1:
                        self.response_cache[i].pop()
                        rewrite_indices.append(i)
                self.action_history[i].append(action)

        # Process Retrieve Document actions
        
        if len(retrieve_indices)>0:
            q_t = [self.construct_query(i) for i in retrieve_indices]
            d_t= self.retrieve(retrieve_indices, q_t)
            for idx, i in enumerate(retrieve_indices):
                self.d_t[i] = d_t[idx]

        # Process Proceed and Rewrite actions in a batch
        batch_indices = proceed_indices + rewrite_indices
        if batch_indices:
            responses = self.get_next_response(batch_indices)
            for idx, i in enumerate(batch_indices):
                self.hat_y_t[i] = responses[idx]
                self.response_cache[i].append(self.hat_y_t[i])

        rewards = self.compute_reward()
        for i in range(self.batch_size):
            next_states.append(self.get_state(i))
            self.steps[i] += 1
            if sum(map(len, self.response_cache[i][1:]))>=256:
                self.done[i]=True
        self.last_action = self.actions.clone()

        return next_states, rewards, self.done, {}

    def compute_reward(self):
        return [0]*self.batch_size
    
    def get_next_response(self, indices):
        response = [self.cat_response(self.response_cache[i]) for i in indices]
        messages = [" ".join(self.collate.templete(self.x[i], response[idx])) for idx, i in enumerate(indices)]#According to the knowledge provided,
        d_t = self.ret.tokenizer.batch_decode([self.d_t[i] for i in indices], skip_special_tokens=True)
        doc_token = self.ret.tokenizer(d_t, return_tensors="pt", padding=True).to(self.LM.device)
        # messages = [messages[j].replace(" </knowledge>", d_t[j]+" </knowledge>") for j in range(len(messages))]
        responses = self.LM.generate(messages, Doc_tokens=doc_token, max_new_tokens=self.step_size, decode=False)
        if self.step_size>64:
            for i in indices:
                self.done[i] = True
        return responses
class Orginal_Env(LLMEnv_test):
    def get_next_response(self, indices):
        response = [self.cat_response(self.response_cache[i]) for i in indices]
        messages = [" ".join(self.collate.templete(self.x[i], response[idx])) for idx, i in enumerate(indices)]
        d_t = self.ret.tokenizer.batch_decode([self.d_t[i] for i in indices], skip_special_tokens=True)
        messages = [messages[j].replace(" </knowledge>", d_t[j]+" </knowledge>") for j in range(len(messages))]
        responses = self.LM.generate(messages, Doc_tokens=None, max_new_tokens=self.step_size, decode=False)
        if self.step_size>64:
            for i in indices:
                self.done[i] = True
        return responses
    
    
class BertAgentCritic(nn.Module):
    def __init__(self, model_config, action_space_size):
        super(BertAgentCritic, self).__init__()
        self.bert = RobertaModel.from_pretrained(config.roberta_dir)
        embedding = self.bert.embeddings.position_embeddings

        self.max_token_length = config.agent_size_config.max_position_embeddings
        new_embedding = torch.nn.Embedding(self.max_token_length +2, embedding.embedding_dim)
        new_embedding.weight.data[:min(len(embedding.weight), new_embedding.num_embeddings),:]=embedding.weight.data[:min(len(embedding.weight), new_embedding.num_embeddings)]
        self.bert.embeddings.position_embeddings = new_embedding
        self.bert.embeddings.register_buffer(
            "position_ids", torch.arange(self.max_token_length +2).expand((1, -1)), persistent=False
        )
        self.bert.embeddings.register_buffer(
            "token_type_ids", torch.zeros(self.bert.embeddings.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(config.roberta_dir)
        self.tokenizer.model_max_length = self.max_token_length
        self.action_head = nn.Linear(self.bert.config.hidden_size, action_space_size)
        self.value_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.action_space_size = action_space_size
        # Add special tokens to the tokenizer
        special_tokens_dict = {'additional_special_tokens': ['[actor_head]', '[value_head]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(['[actor_head]', '[value_head]'])
        self.special_tokens = torch.tensor(self.special_tokens).unsqueeze(0)
    
    @torch.autocast("cuda", torch.bfloat16)
    def forward(self, state=None, inputs=None):
        assert (state is not None and inputs is None) or (state is None and inputs is not None)
        if inputs is None:
        # Tokenize the state and the prompt text
            inputs = self.tokenizer(state, return_tensors="pt", padding=True, truncation=True).to(self.bert.device)
            # Combine the prompt text and state input ids and attention masks
        inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
        }
        
        batch_size = inputs['input_ids'].size(0)
        
        
        # Concatenate prompt text, mask tokens, and special tokens with the input sequence
        inputs['input_ids'] = torch.cat([self.special_tokens.to(self.bert.device).repeat(batch_size, 1), inputs['input_ids']], dim=1)[...,:self.max_token_length]
        inputs['attention_mask'] = torch.cat([torch.ones((batch_size, 2), dtype=torch.long).to(self.bert.device), inputs['attention_mask']], dim=1)[...,:self.max_token_length]
        
        outputs = self.bert(**inputs, output_hidden_states = True)
        # Handle special tokens for action and value heads
        actor_head_output = outputs.hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_size)
        value_head_output = outputs.hidden_states[-1][:, 1, :]  # Shape: (batch_size, hidden_size)
        
        action_logits_special = self.action_head(actor_head_output).float()  # Shape: (batch_size, action_space_size)
        state_value_special = self.value_head(value_head_output)[..., 0].float()  # Shape: (batch_size,)

        return action_logits_special, state_value_special

class PPOTrainer:
    def __init__(self, model:BertAgentCritic, optimizer:torch.optim.Optimizer, gamma=0.99, clip_epsilon=0.2, lambd=0.95, update_epochs=4, batch_size=32, grad_step = 4, ):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lambd = lambd
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.grad_step = grad_step
        
        self.action_coef=1
        self.value_coef=2**1
        
        self.max_entr = torch.tensor(2**-3)
        self.min_entr = torch.tensor(2**-15)
        self.entropy_coef=torch.tensor(2**-10)
        self.collate = collate()
        self.sep = self.collate.datatokenizer.sep_token

    def ppo_loss(self, action_logp, action_dist:Categorical, batch_actions, advantages, returns, values):
        # old_log_probs shape: (batch_size,)
        # batch_action shape: (batch_size,)
        # advantages shape: (batch_size,)
        # returns shape: (batch_size,)
        # values shape: (batch_size,)
        new_action_logp = action_dist.log_prob(batch_actions)
        ratios = torch.exp(torch.clamp(new_action_logp - action_logp, -3, 3))  # Shape: (batch_size,)
        
        # need to broadcast `advantages` to match the shape of `ratios`
        
        surr1 = ratios * advantages  # Shape: (batch_size,)
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # Shape: (batch_size,)
        actor_loss = -torch.min(surr1, surr2).mean()  # Shape: scalar

        critic_loss = F.smooth_l1_loss(values, returns, reduction = "mean", beta = 0.1)  # Shape: scalar
        action_entropy:Tensor = action_dist.entropy().mean() #scalar
        log_p_mean = new_action_logp
        return actor_loss, critic_loss, log_p_mean  # Shape: scalar

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
        
        batch_token = self.model.module.tokenizer(batch_states, return_tensors = "pt", padding = True, truncation=True, return_special_tokens_mask =True)
        questions = [batch_states[i].split(self.sep)[1] for i in range(len(batch_states))]
        questions_token = self.model.module.tokenizer(questions, return_tensors = "pt", padding = True, truncation=True, return_special_tokens_mask =True)
        batch_actions = torch.stack(batch_actions)
        batch_old_log_probs = torch.stack(batch_old_log_probs)
        batch_returns = torch.stack(batch_returns)
        batch_advantages = torch.stack(batch_advantages)
        return questions_token, batch_token, batch_actions, batch_old_log_probs, batch_returns, batch_advantages
    
    def inin_loader(self, memory):
        
        old_states, old_actions, old_log_probs, rewards, dones, values = zip(*memory)
        returns = self.compute_gae(rewards, values, dones, next_value=0)
        old_states = old_states # Shape: (memory_size, state_size)
        old_actions = torch.tensor(old_actions, dtype=torch.long)  # Shape: (memory_size,)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)  # Shape: (memory_size,)
        returns = torch.tensor(returns, dtype=torch.float32)  # Shape: (memory_size,)
        values = torch.tensor(values, dtype=torch.float32)  # Shape: (memory_size,)
        torch.save(returns, "save/return.pt")
        torch.save(values, "save/value.pt")
        advantages = returns - values  # Shape: (memory_size,)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        return [*zip(old_states, old_actions, old_log_probs, returns, advantages)]
    
    def update(self, memory, loader=None): 
        if loader is None:
            data = self.inin_loader(memory)
            loader = DataLoader(data, self.batch_size, True, collate_fn=self.f, pin_memory = True, num_workers=0, persistent_workers=False, drop_last=True)
            
        bar = tqdm(total=self.update_epochs*len(loader), ncols=0)
        self.optimizer.zero_grad()
        step = 0
        for _ in range(self.update_epochs):
            for questions_token, batch_token, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
                step+=1
                batch_token = batch_token.to(self.model.device)  # Shape: (batch_size, n)
                batch_actions = batch_actions.to(self.model.device)  # Shape: (batch_size,)
                batch_old_log_probs = batch_old_log_probs.to(self.model.device)  # Shape: (batch_size,)
                batch_returns = batch_returns.to(self.model.device)  # Shape: (batch_size,)
                batch_advantages = batch_advantages.to(self.model.device)  # Shape: (batch_size,)
                action_logits, state_values = self.model(inputs = batch_token)  # logits shape: (batch_size, action_space_size), state_values shape: (batch_size, 1)
                action_dist = Categorical(logits=action_logits)  # Shape: (batch_size,)
                
                # maximize the state's token for query dist
                questions_token = questions_token.to(self.model.device)
                
                actor_loss, value_loss, a_entropy_loss = self.ppo_loss(batch_old_log_probs, action_dist, batch_actions, batch_advantages, batch_returns, state_values)  # Shape: scalar
                a_entropy_loss = a_entropy_loss.clamp(-3.0, 0.).mean()
                if -a_entropy_loss>0.3:
                    self.entropy_coef/=1.05
                else:
                    self.entropy_coef*=1.05
                self.entropy_coef = torch.clamp(self.entropy_coef, self.min_entr, self.max_entr)
                loss:Tensor = self.action_coef*actor_loss+ self.value_coef*value_loss+ self.entropy_coef*a_entropy_loss# + 0.001*query_norm_loss
                loss.backward()
                if step%self.grad_step==0:
                    torch.nn.utils.clip_grad_norm_(chain(*[self.optimizer.param_groups[param_i]['params'] for param_i in [0,1,2]]), 1.0)
                    self.optimizer.step()
                count0 = batch_actions.tolist().count(0)/len(batch_actions)
                count1 = batch_actions.tolist().count(1)/len(batch_actions)
                count2 = batch_actions.tolist().count(2)/len(batch_actions)
                bar.set_postfix_str(f"ac: {actor_loss:6.3f}, value: {value_loss:.3f}, entropy: {-a_entropy_loss:.3f}, div: {count0:.2f}, {count1:.2f}, {count2:.2f}")
                bar.update()


# if __name__=='__main__':
    
    
    # # Example usage
    # env = LLMEnv()
    # model = BertAgentCritic(agent_size_config, env.action_space_size)
    # trainer = PPOTrainer(model)

    # # Training loop
    # for episode in range(1000):
    #     state = env.reset()  # Shape: string
    #     done = False
    #     memory = []

    #     while not done:
    #         token_logits, action_logits, state_value = model([state])  # action_logits shape: (1, action_space_size), state_value shape: (1, 1)
    #         token_dist = Categorical(logits = token_logits)
    #         action_dist = Categorical(logits = action_logits)
    #         tokens = token_dist.sample()
    #         action = action_dist.sample()  # Shape: (1,)
    #         print(model.tokenizer.batch_decode(tokens))
    #         exit()

    #         next_state, reward, done, _ = env.step(action.item())  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
    #         memory.append((state, action, dist.log_prob(action), reward, done, state_value))  # Shapes: (string, (1,), (1, action_space_size), scalar, scalar (boolean), (1, 1))

    #         state = next_state

    #     trainer.update(memory)