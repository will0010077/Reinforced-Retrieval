import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"


import sys
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from tqdm import tqdm


from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DatasetLoader.collate_func import collate
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask, tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder
from train_ret_2 import NQADataset
from metric.reward import metric
import yaml
import peft

from transformers import AutoTokenizer
import config
import numpy as np


token = "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"
bert_dir = "huggingface/bert"
LM_dir = "/usr/model/llama2-7b/"

if __name__=="__main__":
    print(torch.cuda.device_count())
    device='cuda'
    metric_c = metric()
    metric_c.to(device)
    
    
    print('Loading dataset...')
    data_path='data/dev_with_doc.jsonl'
    num_testing=200
    dataset=NQADataset(data_path=data_path, num_samples=1000, use_doc=True)
    
    Enc=True
    Policy=True
    print('Loading LLM')
    generate_config = config.generate_config
    generate_config.temperature=0.2
    if not Policy:
        generate_config.do_sample=False
    LM = LLaMa_reader(LM_dir, device, token = token, from_pretrained=True, generate_config=generate_config)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder=KnowEncoder(dims = num_dims, **config.enc_config, dtype=dtype)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=32, adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    LM.to(device)
    LM.eval()

    if Enc and True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/EncLM_5.pt", map_location='cpu'))
    # init retriever

    print('Initilize retriever')
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.to(device)
    lex_MAE_retriver.eval()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)
    
    

    max_epoch = 10
    num_retrieve=1
    num_neg=16
    num_RL_update = 8

    env_bs=8
    if Enc:
        env = LLMEnv_test(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs, shuffle=False, step_size=15 if Policy else 256)
    else:
        env = Orginal_Env(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs, shuffle=False, step_size=15 if Policy else 256)
    
    print("Initialize Agent...")
    agent = BertAgentCritic(config.agent_size_config, env.action_space_size).to(torch.bfloat16)
    agent.to(device)
    agent.eval()
    agent.load_state_dict(torch.load("save/Agent_0.pt", map_location="cpu"))
    
    # Training loop
    total = 100000
    memory = []
    ma_reward=0.
    episode=0
    done = [True]*env_bs
    state=[None]*env_bs
    q_list=[]
    a_list=[]
    true_list=[]
    print("Starting reset...")
    f = open("moniter.txt", "a")
    
    for i in range(env_bs):
        if done[i]:
            state[i] = env.reset(i)  # Shape: string
            done[i]=False
    while True:
        for i in range(env_bs):
            if done[i]:
                q_list.append(env.x[i])
                a_list.append(env.cat_response(env.response_cache[i]))
                true_list.append(" ".join(env.y[i]))
                episode+=1
                state[i] = env.reset(i)  # Shape: string
                done[i]=False
        if len(q_list)>=num_testing:
            break
        while not any(done):
            with torch.no_grad():
                action_logits, state_value = agent(state)  # token_logits:(B, num, vocab), action_logits shape: (B, action_space_size), state_value shape: (B,)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            action_logits[:,1]-=0.5
            action_dist = Categorical(logits = action_logits/0.1)
            action = action_dist.sample()  # Shape: (B,)
            if not Policy:
                action[:]=1
            print(action[0].item(), end='', flush=True)
            
            next_state, reward, done, _ = env.step(action)  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            # print(env.cat_response(env.response_cache))
            state = next_state
            
            
    if Enc:
        env = LLMEnv_test(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs, shuffle=False, step_size=256)
    else:
        env = Orginal_Env(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs, shuffle=False, step_size=256)
            
    q_list2=[]
    a_list2=[]
    true_list2=[]
    done = [True]*env_bs
    for i in range(env_bs):
        if done[i]:
            state[i] = env.reset(i)  # Shape: string
            done[i]=False
    while True:
        for i in range(env_bs):
            if done[i]:
                q_list2.append(env.x[i])
                a_list2.append(env.cat_response(env.response_cache[i]))
                true_list2.append(" ".join(env.y[i]))
                episode+=1
                state[i] = env.reset(i)  # Shape: string
                done[i]=False
        if len(q_list2)>=num_testing+16:
            break
        while not any(done):
            with torch.no_grad():
                action_logits, state_value = agent(state)  # token_logits:(B, num, vocab), action_logits shape: (B, action_space_size), state_value shape: (B,)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            action_logits[:,1]-=0.5
            action_dist = Categorical(logits = action_logits/0.1)
            action = action_dist.sample()  # Shape: (B,)

            action[:]=1
            print(action[0].item(), end='', flush=True)
            
            next_state, reward, done, _ = env.step(action)  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            # print(env.cat_response(env.response_cache))
            state = next_state
            
    bert = metric_c.Bert_score(a_list, true_list )
    R_1, R_2, R_L = metric_c.ROUGE_score(a_list, true_list )
    bleu = metric_c.BLEU_1_score(a_list, true_list)
    
    bert2 = metric_c.Bert_score(a_list2, true_list2 )
    R_12, R_22, R_L2 = metric_c.ROUGE_score(a_list2, true_list2 )
    bleu2 = metric_c.BLEU_1_score(a_list2, true_list2)
    inlist = []
    for j in range(len(q_list)):
        for i in range(len(q_list2)):
            if q_list2[i] == q_list[j]:
                break
        inlist.append(i)
        f.write(
f'''Prompt: {q_list[j]}\nGround truth: {true_list[j]}
[{bleu[j]:.3f}, {R_1[j]:.3f}, {R_2[j]:.3f}, {R_L[j]:.3f}, {bert[j]:.3f}] Agent refined response: {a_list[j]}
[{bleu2[i]:.3f}, {R_12[i]:.3f}, {R_22[i]:.3f}, {R_L2[i]:.3f}, {bert2[i]:.3f}] Ori response: {a_list2[i]}
''' +"="*80+"\n")
        
    f.write(f"Enc:{Enc}, Policy:{Policy}\n")
    f.write(f"BLEU_1: {sum(bleu)/len(bleu)*100:05.2f}\n")
    f.write(f"ROUGE-1: {sum(R_1)/len(R_1)*100:05.2f}\n")
    f.write(f"ROUGE-2: {sum(R_2)/len(R_2)*100:05.2f}\n")
    f.write(f"ROUGE-L: {sum(R_L)/len(R_L)*100:05.2f}\n")
    f.write(f"BERT: {sum(bert)/len(bert)*100:05.2f}\n")
    f.write(f"Without Policy\n")
    f.write(f"BLEU_1: {sum([bleu2[i] for i in inlist])/len(inlist)*100:05.2f}\n")
    f.write(f"ROUGE-1: {sum([R_12[i] for i in inlist])/len(inlist)*100:05.2f}\n")
    f.write(f"ROUGE-2: {sum([R_22[i] for i in inlist])/len(inlist)*100:05.2f}\n")
    f.write(f"ROUGE-L: {sum([R_L2[i] for i in inlist])/len(inlist)*100:05.2f}\n")
    f.write(f"BERT: {sum([bert2[i] for i in inlist])/len(inlist)*100:05.2f}\n")