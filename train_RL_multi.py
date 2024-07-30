import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"


import sys
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch import optim 
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Manager

from RL.utils import BertAgentCritic, PPOTrainer, LLMEnv_batch_version
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DatasetLoader.collate_func import collate
from DatasetLoader.dataset import NQADataset
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask, tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder
from metric.reward import BLEU_1_score, Bert_score
import yaml
import peft
from time import time

from transformers import AutoTokenizer
import config


token = "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"
def doc_templete(doc:list[str]):
    return  '\n\n'.join(doc)
def templete(doc_list:list[str], query:str, answer:str)->tuple[str]:
    doc_list = doc_templete(doc_list)
    prompt = f'''<<SYS>>\n This is the searched knowledge: [KNOW] {doc_list} [/KNOW]
    Please answer user questions based on the above knowledge\n<</SYS>>
    \n [INST] User: {query.strip()} [/INST] Assistant: '''
    return prompt, prompt + answer
def prepare_QA_token(tokenizer, doc:list[list[str]], texts:list[str], targets:list[str]):
    '''
    
    '''
    
    unlabel, cat_qa = zip(*[templete(doc_list, q, a) for doc_list, q,a in zip(doc, texts, targets)])
    question_str = unlabel
    unlabel = tokenizer(text=unlabel).input_ids
    # print(max([len(s) for s in unlabel]))
    tokens = tokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=128, truncation =True,)
    
    for i in range(len(texts)):
        tokens['labels'][i, :len(unlabel[i])]=-100
    tokens['labels'][tokens['attention_mask']==0]=-100
    return tokens, question_str

def state_template(query:list[str], generation:list[str], doc:list[str]):
    context = [f'''{doc_templete(doc_list)} in this article. question "{q}". The answer is: "{a}" the query is: \"''' for doc_list, q,a in zip(doc, query, generation)]

    return context


def collect_trajectory(env, agent, trajectory, memory, done, state, env_bs):
    for i in range(env_bs):
        if done[i]:
            state[i] = env.reset(i)  # Shape: string
            done[i] = False
    while not any(done):
        with torch.no_grad():
            action_logits, state_value = agent(state)  # action_logits shape: (B, action_space_size), state_value shape: (B,)
        action_logits, state_value = action_logits.cpu(), state_value.cpu()
        
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()  # Shape: (B,)
        if torch.rand([1]) < 0.05:
            action = torch.randint(env.action_space_size, [env_bs])
        else:
            action = action_dist.sample()  # Shape: (B,)

        next_state, reward, done, _ = env.step(action)  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)

        action_logp = action_dist.log_prob(action)
        for i in range(env_bs):
            trajectory[i].append([state[i], action[i], action_logp[i], reward[i], done[i], state_value[i]])  # Shapes: (string, (1,), (1),(15), (15) scalar, scalar (boolean), (1, 1))
        state = next_state
    for i in range(env_bs):
        if done[i]:
            for j in range(env.steps[i]):
                trajectory[i][j][3] = env.revise_reward[i][j]
            memory.extend(trajectory[i])
            trajectory[i] = []
    return memory, state, done

def training(rank, world_size, total, shared_memory, dataset, LM, lex_MAE_retriver, agent):
    import torch.distributed as dist

    print(f"Running DDP on rank {rank}.")

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    env_bs = 64
    env = LLMEnv_batch_version(dataset, LM, lex_MAE_retriver, 3, batch_size=env_bs)
    Agent_optim = optim.AdamW([{"params": agent.bert.parameters(), "lr": config.train_config.agent_lr},
                               {"params": agent.value_head.parameters(), "lr": config.train_config.agent_lr*3},
                               {"params": agent.action_head.parameters(), "lr": config.train_config.agent_lr*3}], betas=config.train_config.betas, eps=1e-4)
    
    trainer = PPOTrainer(agent, Agent_optim, gamma=1.0, clip_epsilon=0.2, lambd=0.95, update_epochs=8, batch_size=128, grad_step=1)
    
    total = 100000
    reduce = optim.lr_scheduler.PolynomialLR(Agent_optim, total_iters=int(total*1.2), power=1.5)
    warmup = optim.lr_scheduler.LinearLR(Agent_optim, 1e-5, 1, total_iters=int(total*0.001))
    scheduler = optim.lr_scheduler.SequentialLR(Agent_optim, [warmup, reduce], milestones=[warmup.total_iters])
    
    memory = []
    ma_reward = 0
    reward_file = open(f"reward_number_{rank}.txt", "a")
    print("Start training...")
    
    trajectory = [[] for _ in range(env_bs)]  # don't do this->[[]]*env_bs
    done = [True]*env_bs
    state = [None]*env_bs
    episode = 0
    save_time = time()
    all_memory = []
    
    while True:
        memory, state, done = collect_trajectory(env, agent, trajectory, memory, done, state, env_bs)
        shared_memory.extend(memory)
        memory = []

        if rank == 0:
            if len(shared_memory) > 2048:
                reward_file.flush()

                temp_memory = list(shared_memory)
                shared_memory.clear()
                trainer.update(temp_memory)

                # Broadcast the updated model to other ranks
                for param in agent.parameters():
                    dist.broadcast(param.data, src=0)
                    
        else:
            if rank != 0:
                # Receive the updated model from rank 0
                for param in agent.parameters():
                    dist.broadcast(param.data, src=0)
        
        if time() - save_time > 29.5*60:
            if rank == 0:
                # Save Agent weight
                torch.save(agent.state_dict(), f"./save/Agent_{rank}.pt")
                save_time = time()

def main():
    world_size = 4  # Number of processes to run in parallel
    total = 100000

    print(torch.cuda.device_count())
    device = 'cuda'

    print('Loading LLM')
    LM = LLaMa_reader(config.LM_dir, device, token=config.token, from_pretrained=True)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder = KnowEncoder(dims=num_dims, **config.enc_config, dtype=dtype)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=config.enc_config.num_layers, adapter_len=1)}
    LM = EncTunedLM(LM, Enc=Encoder, configs=peft_configs, adapter_name='Enc')
    LM.to(device)
    LM.eval()

    print(f'Loading EncTunedLM weight...')
    LM.load_state_dict(torch.load("save/EncLM.pt", map_location='cpu'))

    print('Initialize retriever')
    lex_MAE_retriver = lex_retriever()
    lex_MAE_retriver.to(device)
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)
    lex_MAE_retriver.eval()
    
    print('Loading dataset...')
    data_path = 'data/cleandata_with_doc.jsonl'
    dataset = NQADataset(data_path=data_path, num_samples=None, use_doc=True)
    
    print("Initialize Agent...")
    agent = BertAgentCritic(config.agent_size_config, env.action_space_size).to(torch.bfloat16)
    agent.to(device)
    agent.share_memory()

    shared_memory = Manager().list()
    mp.spawn(training,
             args=(world_size, total, shared_memory, dataset, LM, lex_MAE_retriver, agent),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
    
            
        





