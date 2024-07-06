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
from fintune_contriver import NQADataset
from metric.reward import BLEU_score, Bert_score
import yaml
import peft

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
if __name__=="__main__":
    print(torch.cuda.device_count())
    device='cuda'
    
    cluster_config=config.cluster_config
    cluster = cluster_builder(k=cluster_config.k)
    cluster.load('05_29_14_30')

    print('Loading LLM')
    LM = LLaMa_reader(config.LM_dir, device, token = token, from_pretrained=True)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    print(f'Initialize KnowEnc with {dtype}...')
    Encoder=KnowEncoder(dims = num_dims, **config.enc_config, dtype=dtype)

    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=config.enc_config.num_layers, adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    LM.to(device)
    LM.eval()

    if True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/EncLM.pt", map_location='cpu'))
    # init retriever

    print('Initilize retriever')
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=True)
    data=torch.load('data/data_reduced_2000000.pt') ## shape:(N,d)
    retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
    retriever.to(device)
    retriever.model.to(device)
    retriever.model.device=device
    del lex_MAE_retriver, data, cluster
    
    
    print("Initialize Agent...")
    
    # RL_bs = 64
    # config = PPOConfig(
    #     model_name="gpt2",
    #     learning_rate=1.e-5,
    #     mini_batch_size = 16,
    #     batch_size = RL_bs,
    #     gradient_accumulation_steps = 4,
    # )
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, device_map=device)
    # model.to(device)
    # gpt2tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # gpt2tokenizer.pad_token = gpt2tokenizer.eos_token
    # gpt2tokenizer.padding_side = "left"
    
    # ppo_trainer = PPOTrainer(
    #     model=model,
    #     config=config,
    #     tokenizer=gpt2tokenizer,
    # )
    # stop_id = [gpt2tokenizer.eos_token_id]
    # for k,v in gpt2tokenizer.vocab.items():
    #     if "\"" in k:
    #         stop_id.append(v)
    # generation_kwargs = {
    #     "min_length": -1,
    #     "top_k": 30,
    #     "top_p": 0.7,
    #     "do_sample": True,
    #     "pad_token_id": gpt2tokenizer.eos_token_id,
    #     "no_repeat_ngram_size" : 4, 
    #     "eos_token_id" : stop_id,
    #     "max_new_tokens" : 16,
    # }
    
    # Example usage
    max_epoch = 10
    num_retrieve=1
    num_neg=16
    num_RL_update = 8

    print('Loading dataset...')
    data_path='data/cleandata.jsonl'
    dataset=NQADataset(data_path=data_path)
    
    env = LLMEnv(dataset, LM, retriever, 3)
    agent = BertAgentCritic(config.agent_size_config, env.action_space_size).to(torch.bfloat16)
    # agent.load_state_dict(torch.load("./save/Agent.pt"),strict=False)
    agent.to(device)
    
    Agent_optim = optim.AdamW([{"params": agent.bert.parameters(), "lr": config.train_config.agent_lr},
                               {"params": agent.value_head.parameters(), "lr": config.train_config.agent_lr*3},
                               {"params": agent.action_head.parameters(), "lr": config.train_config.agent_lr*3}], betas = [0.8, 0.98], eps=1e-4)
    trainer = PPOTrainer(agent, Agent_optim, gamma = 0.99,  lambd = 0.96, update_epochs=4, batch_size = 64, grad_step = 1)
    # Training loop
    total = 100000
    scheduler = optim.lr_scheduler.PolynomialLR(Agent_optim, total_iters=int(total*1.2), power = 1.5)
    memory = []
    ma_reward=0
    reward_file = open("reward_number.txt", "w")
    for episode in range(total):
        state = env.reset()  # Shape: string
        done = False
        reward_list = []
        while not done:
            with torch.no_grad():
                action_logits, state_value = agent([state])  # action_logits shape: (1, action_space_size), state_value shape: (1, 1)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            dist = Categorical(logits = action_logits)
            if torch.rand([1])<0.05 :
                action = torch.randint(env.action_space_size, [1])
            else:
                action = dist.sample()  # Shape: (1,)
            
            # if episode%20==0:
            #     action = torch.tensor(env.steps%3)
            print(action.item(), end='', flush=True)
            next_state, reward, done, _ = env.step(action.item())  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            if reward!=reward: # check nan, reach max_len of LLM, the output is empty
                print("NAN!!!!")
                env.revise_reward.pop()
                env.steps -= 1
                break
            memory.append([state, action, dist.log_prob(action), reward, done, state_value])  # Shapes: (string, (1,), (1, action_space_size), scalar, scalar (boolean), (1, 1))
            reward_list.append(reward)
            state = next_state
        # modify memory with revise reward that consider future
        for i in reversed(range(env.steps)):
            memory[-i-1][3] = env.revise_reward[-i-1]
        # print("\r"," "*80,"\r", end='\n')
        # print(env.cat_response(env.response_cache))
        ma_reward = 0.95*ma_reward + 0.05*sum(env.revise_reward)
        reward_file.write(f"{sum(env.revise_reward):.5f}\n")
        print("\nreward: ",ma_reward, end="\r")
        if len(memory)>(512):
            reward_file.flush()
            print()
            trainer.update(memory)
            memory = []
        scheduler.step()
        if (episode+1)%2000==0:
            #save Agent weight
            torch.save(agent.state_dict(), "./save/Agent.pt")
    exit()
    
    
            
        





