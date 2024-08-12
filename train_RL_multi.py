import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import sys
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch import optim 
from tqdm import tqdm
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from socket import socket
from functools import partial

from RL.utils import BertAgentCritic, PPOTrainer, LLMEnv_batch_version
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DatasetLoader.collate_func import collate
from DatasetLoader.dataset import NQADataset
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask, tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder
import yaml
import peft
from time import time

from transformers import AutoTokenizer
import config


token = "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"

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker_init(rank, world_size, port, x):
    setup(rank, world_size, port)
def training(rank:int, world_size:int, port, total:int, env:LLMEnv_batch_version, agent:BertAgentCritic, memory):
    import torch.distributed as dist

    print(f"Running on rank {rank}.")
    setup(rank, world_size, port)


    env.to(rank)
    ratio = 1.
    agent = agent.to(rank)
    agentDDP = DDP(agent, device_ids=[rank], find_unused_parameters=True)
    Agent_optim = optim.AdamW([{"params": agentDDP.module.bert.parameters(), "lr": config.train_config.agent_lr* ratio},
                               {"params": agentDDP.module.value_head.parameters(), "lr": config.train_config.agent_head_lr* ratio, "weight_decay": 0.02},
                               {"params": agentDDP.module.action_head.parameters(), "lr": config.train_config.agent_head_lr * ratio, "weight_decay": 0.02}], betas=config.train_config.betas)
    
    trainer = PPOTrainer(agentDDP, Agent_optim, update_epochs=max(4//world_size, 1), **config.ppo_config)
    
    reduce = optim.lr_scheduler.PolynomialLR(Agent_optim, total_iters=int(total*1.2), power=1.5)
    warmup = optim.lr_scheduler.LinearLR(Agent_optim, 1e-5, 1, total_iters=int(total*0.001))
    scheduler = optim.lr_scheduler.SequentialLR(Agent_optim, [warmup, reduce], milestones=[warmup.total_iters])
    
    
    ma_reward = 0
    reward_file = open(f"reward_number_{rank}.log", "a")
    print("Start training...")
    
    trajectory = [[] for _ in range(env.batch_size)]  # don't do this->[[]]*env_bs
    done = [True]*env.batch_size
    state = [None]*env.batch_size
    save_time = time()
    
    while True:
        for i in range(env.batch_size):
            if done[i]:
                state[i] = env.reset(i)  # Shape: string
                done[i] = False
                scheduler.step()
        while not any(done):
            with torch.no_grad():
                action_logits, state_value = agent(state)  # action_logits shape: (B, action_space_size), state_value shape: (B,)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()  # Shape: (B,)
            if torch.rand([1]) < 0.05:
                action = torch.randint(env.action_space_size, [env.batch_size])
            else:
                action = action_dist.sample()  # Shape: (B,)

            next_state, reward, done, _ = env.step(action)  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            # print(action, reward, done)
            action_logp = action_dist.log_prob(action)
            for i in range(env.batch_size):
                trajectory[i].append([state[i], action[i], action_logp[i], reward[i], done[i], state_value[i]])  # Shapes: (string, (1,), (1),(15), (15) scalar, scalar (boolean), (1, 1))
            state = next_state
            
            
        rewards = []
        for i in range(env.batch_size):
            if done[i]:
                traj_reward = [trajectory[i][j][3] for j in range(env.steps[i])]
                rewards.append(sum(traj_reward))
                memory.extend(trajectory[i])
                trajectory[i] = []
                # print(env.cat_response(env.response_cache[i]))

        for r in rewards:
            reward_file.write(f"{r:.5f}\n")
            
        if len(memory)>(512):
            dist.barrier()
            reward_file.flush()
            data = trainer.inin_loader(memory)
            loader = DataLoader(data, trainer.batch_size, True, collate_fn=trainer.f, pin_memory = True, num_workers=0, persistent_workers=False, drop_last=True)
            
            trainer.update(memory, loader)
            dist.barrier()
            del loader
            if rank==0:
                memory[:] = []

        if time() - save_time > 29.5*60:
            save_time = time()
            if rank == 0:
                # Save Agent weight
                torch.save(agentDDP.module.state_dict(), f"./save/Agent_{rank}.pt")

def main():
    world_size = torch.cuda.device_count()
    # world_size = 1
    total = 5000
    env_bs = 32

    print(torch.cuda.device_count())

    print('Loading LLM')
    LM = LLaMa_reader(config.LM_dir, "cpu", token=config.token, from_pretrained=True)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder = KnowEncoder(dims=num_dims, **config.enc_config, dtype=dtype).to(torch.bfloat16)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=32, adapter_len=1)}
    LM = EncTunedLM(LM, Enc=Encoder, configs=peft_configs, adapter_name='Enc')
    LM.eval()

    print(f'Loading EncTunedLM weight...')
    LM.load_state_dict(torch.load("save/EncLM_5.pt", map_location='cpu'))

    print('Initialize retriever')
    lex_MAE_retriver = lex_retriever()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)
    lex_MAE_retriver.eval()
    
    
    print('Loading dataset...')
    data_path = 'data/cleandata_with_doc.jsonl'
    dataset = NQADataset(data_path=data_path, num_samples=18, use_doc=True)
    
    env = LLMEnv_batch_version(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs)
    
    print("Initialize Agent...")
    agent = BertAgentCritic(config.agent_size_config, 3).to(torch.bfloat16)
    agent.load_state_dict(torch.load("save/Agent_0.pt", map_location="cpu"))
    memory = multiprocessing.Manager().list() 
    with socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    mp.spawn(training,
             args=(world_size, port, total, env, agent, memory),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
    
            
        





