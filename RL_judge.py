import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"


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
bert_dir = "huggingface/bert"
LM_dir = "/usr/model/llama2-7b/"
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
    LM = LLaMa_reader(LM_dir, device, token = token, from_pretrained=True)
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
    
    env = CustomEnv(dataset, LM, retriever, 4)
    agent = BertAgentCritic(config.agent_size_config, env.action_space_size)
    agent.to(device)
    
    Agent_optim = optim.AdamW(agent.parameters(), lr = 1e-5)
    trainer = PPOTrainer(agent, Agent_optim, update_epochs=4)
    # Training loop
    memory = []
    ma_reward=0.5
    for episode in range(100000):
        state = env.reset()  # Shape: string
        done = False
        reward_list = []
        while not done:
            with torch.no_grad():
                action_logits, state_value = agent([state])  # action_logits shape: (1, action_space_size), state_value shape: (1, 1)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            action_prob = torch.softmax(action_logits, dim=-1)  # Shape: (1, action_space_size)
            dist = Categorical(action_prob)
            if torch.rand([1])<0.05:
                action = torch.randint(env.action_space_size, [1])
            else:
                action = dist.sample()  # Shape: (1,)
            print(action.item(), end='', flush=True)
            next_state, reward, done, _ = env.step(action.item())  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            memory.append((state, action, dist.log_prob(action), reward, done, state_value))  # Shapes: (string, (1,), (1, action_space_size), scalar, scalar (boolean), (1, 1))
            reward_list.append(reward)
            state = next_state
        print("\r"," "*40,"\r", end='')
        ma_reward = 0.95*ma_reward + 0.05*sum(reward_list)
        print("reward: ",ma_reward, end="\r")
        if episode%16==0:
            trainer.update(memory)
            memory=[]
    exit()
    
    
    print('Loading dataset...')
    data_path='data/cleandata.jsonl'
    dataset=NQADataset(data_path=data_path)
    # dataset.data=dataset.data[:5]*10000
    loader = DataLoader(dataset, batch_size = 1, shuffle=True)
    
    
    collate_fn = collate(LM_dir, bert_dir)
    ma_loss=10
    ma_reward=-2
    iter=0
    
    for epoch in range(max_epoch):
        train_bar=tqdm(loader, ncols=0)
        stream = torch.cuda.current_stream()
        qry_buffer, res_buffer, re_buffer = [],[],[]
        
        for query, target in train_bar:
            B = len(query)
            
            
            # send to gpu to retrieval loop
            doc_set = ['']*len(query)
            # a = Agent(q, d, y)
            #target = target.split()
            
            for i, y in enumerate(target):
                # Initialize
                y:list[str] = y.split(' ')
                chunk_size = 10
                y = [' '.join(y[i:i+chunk_size]) for i in range(0,len(y), chunk_size)]
                messages = state_template([query[i]], [target[i]], [''])
                res_cache = ['']
                rewards=[]
                n = -1
                # RL loop
                T=len(y)
                for t in range(T):
                    # select action
                    query_tensors = gpt2tokenizer(messages, max_length=256, truncation=True).input_ids
                    query_tensors = [torch.tensor(q) for q in query_tensors]
                    #### Get response from SFTModel
                    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
                    response = [gpt2tokenizer.decode(r[len(q):],skip_special_tokens=True) for q, r in zip(query_tensors, response_tensors)]
                    print(response)
                    #### Retrieve
                    dt, zt = retriever.retrieve(query[i]+' '.join(res_cache), k=1, num_search=4)
                    dt = dt.squeeze(1)
                    dt = tensor_retuen_type(input_ids = dt, attention_mask = torch.ones_like(dt)).to(device)
                    
                    #### Take the action and Compute reward
                    n = n+1
                    qa_tokens, unlabel, unlabel_str, q_str, a_str, a_tokens = collate_fn.collate_qa([(query[i], y[i])])
                    p_generation = LM.pseudo_generate(unlabel_str[0]+' '.join(res_cache), y[n], Doc_tokens = dt)
                    res_cache += p_generation
                    pseudo_bert=Bert_score(p_generation, [y[n]])
                    qry_buffer.extend(query_tensors)
                    res_buffer.extend(response_tensors)
                    re_buffer.extend(pseudo_bert)
                
                # update replay buffer
                
            
            #### Run PPO step
            train_bar.set_postfix_str(f'len: {len(re_buffer)}, reward: {sum(re_buffer)/len(re_buffer):.3f}')
            if len(re_buffer)>=RL_bs:
                stats = ppo_trainer.step(qry_buffer[:RL_bs], res_buffer[:RL_bs], re_buffer[:RL_bs])
                qry_buffer, res_buffer, re_buffer = qry_buffer[RL_bs:], res_buffer[RL_bs:], re_buffer[RL_bs:]
            
            # END of T
                    

                

            
            # policy.to('cpu')
            # retriever.to('cpu')
            
            
            
            # print(ret.shape, outputs.shape, doc_set.shape)#torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, n])
            
            # doc = [retriever.tokenizer.batch_decode(doc_set[i]) for i in range(len(doc_set))]

            # if iter%100==0:
            #     with open("moniter.txt", 'a') as f:
            #         f.write(question_str[0] + LM.generate(question_str[0], prefix = [p[0:1] for p in prefix])[0] + '\n')
            #         f.write('Ground Truth: '+ target[0])
                    
                
            iter+=1
            
                    

            
        





