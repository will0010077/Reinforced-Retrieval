import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask
from LM.llama_reader import LLaMa_reader, EncoderAdaptedModel
from LM.Knowledge_encoder import KnowEncoder
from fintune_contriver import NQADataset
import yaml
import peft

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

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
    context = [f'''**Document Context**:
```
{doc_templete(doc_list)}
```

**User Question**:
```
{q}
```

**Generated Answer**:
```
{a}
```

**Instruction**:
- Carefully read the provided document.
- Review the generated answer in the context of the document and the user question.
- Indicate whether the generated answer should be rewritten or not.

The generated answer need to be rewritten(True/False):''' for doc_list, q,a in zip(doc, query, generation)]

    return context
if __name__=="__main__":
    device='cuda'
    
    cluster_config=config["cluster_config"]
    cluster = cluster_builder(k=cluster_config["k"])
    cluster.load('05_29_14_30')

    # init module
    # encoder:torch.Module
    print('Loading LLM')
    adapter_layers=8
    LM = LLaMa_reader("MediaTek-Research/Breeze-7B-Instruct-v0_1", device, adapter_layers=adapter_layers)
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    
    
    print('Initialize KnowEnc...')
    Encoder=KnowEncoder(adapter_layers, num_dims, config['train_config']['head'], num_prefix=16)
    pretrain_Encoder:dict = torch.load('save/LEX_MAE_retriever_loss_6.8032.pt', map_location='cpu')['dec_model_state_dict']
    for k in [*pretrain_Encoder.keys()]:
        if 'predictions' not in k:
            pretrain_Encoder['.'.join(k.split('.')[2:])] = pretrain_Encoder[k]
        del pretrain_Encoder[k]
    pretrain_Encoder['pooler.dense.weight'] = Encoder.model.pooler.dense.weight
    pretrain_Encoder['pooler.dense.bias'] = Encoder.model.pooler.dense.bias
    Encoder.model.load_state_dict(pretrain_Encoder, assign=True)
    Encoder.to(device)
    del pretrain_Encoder
    Enc_optim = torch.optim.AdamW([*Encoder.parameters()]+[p for p in LM.model.parameters() if p.requires_grad], lr = config['train_config']['enc_lr'])
    
    print('testing prefix tuning...')
    prefix = Encoder.forward(['hello'], k = 1, dtype=torch.float16, device = device)
    tokens = LM.tokenizer(['world'], return_tensors='pt').to(device)
    print(LM.forward(**tokens, prefix = prefix).logits.shape)
    print(LM.generate(["Hello, I'm", "Transformer is a framwork in NLP"], max_new_tokens=32, prefix = prefix))

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
    
    
    max_epoch = 10
    num_retrieve=1
    num_neg=16
    num_RL_update = 8

    
    print('Loading dataset...')
    data_path='data/cleandata.pt'
    dataset=NQADataset(data_path=data_path)
    # dataset.data=dataset.data[:5]*10000
    loader = DataLoader(dataset, batch_size = 16, shuffle=True)
    
    ma_loss=10
    ma_reward=-2
    iter=0
    
    for epoch in range(max_epoch):
        train_bar=tqdm(loader, ncols=0)
        stream = torch.cuda.current_stream()
        for q, target in train_bar:
            B = len(q)
            
            
            # send to gpu to retrieval loop
            doc_set = []
            # a = Agent(q, d, y)
            action=['retrieve', 'generate']*5
            st={"Q":q, "A":target, "y": ['']*len(q)}
            for t in range(4):
                st_ = st
                a=action[t]
                match a:
                    case 'retrieve':
                        with torch.no_grad():
                            dt, zt = retriever.retrieve(q, k=num_retrieve, num_search=4)# [B, neg, n] [B, neg, 30522]
                            # pos neg pair reshape
                            doc_set = dt
                            doc_set = doc_set.reshape([-1, doc_set.shape[-1]])#(B*k, n)
                            st_['d'] = doc_set

                    case 'generate':
                        # feed doc into KnowEnc to get prefix
                        if config['train_config']['use_prefix']:
                            mask = generate_mask(st['d'], Encoder.tokenizer.pad_token_id)
                            tokens = tensor_retuen_type(input_ids = st_['d'], attention_mask = mask).to(device)
                            prefix = Encoder.forward(tokens, k = num_retrieve, dtype=torch.float16)
                            del tokens
                        else:
                            prefix = None
                        # QA pre-process
                        tokens, question_str = prepare_QA_token(LM.tokenizer, [[]]*B, q, target)
                        tokens = tokens.to(device)
                        labels = tokens['labels']
                        
                        
                        # !!!LLM prefix tuning forward, loss and reward!!!
                        # y, loss = LM.forward(**tokens, prefix=prefix)
                        # del y
                        LM_output = LM.forward(**tokens, prefix=prefix)
                        loss = LM_output.loss
                        del LM_output.logits
                        reward = -loss.detach()# temperaly, (B)
                        loss = loss.mean()
                        prefix[0].retain_grad()
                        if config['train_config']['use_prefix']:
                            Enc_optim.zero_grad()
                            loss.backward()
                            Enc_optim.step()

                        
                        cat_input = [a+b+' ' for a,b in zip(question_str, st_['y'])]
                        # print('cat input',cat_input)
                        yt_ = LM.generate(cat_input, prefix=prefix, stop_strings=['.'])
                        
                        st_['y'] = [a+' '+b for a,b in zip(st_['y'], yt_)]
                    case _:
                        pass
                st = st_
            # END of T
                    

                

            
            # policy.to('cpu')
            # retriever.to('cpu')
            
            
            
            # print(ret.shape, outputs.shape, doc_set.shape)#torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret+1, 30522]) torch.Size([8, #ret, n])
            
            # doc = [retriever.tokenizer.batch_decode(doc_set[i]) for i in range(len(doc_set))]

            if iter%10==0:
                train_bar.set_postfix_str(f'len: {tokens.input_ids.shape[-1]}, loss: {loss.item():.3f}, reward: {ma_reward:.2f}')
            if iter%100==0:
                with open("moniter.txt", 'a') as f:
                    f.write(question_str[0] + LM.generate(question_str[0], prefix = [p[0:1] for p in prefix])[0] + '\n')
                    f.write('Ground Truth: '+ target[0])
                    
                
            iter+=1
            
                    

            
        





