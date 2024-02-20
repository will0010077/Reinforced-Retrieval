from Knowledge_encoder import KnowEncoder
from llama_reader import LLaMa_reader
from Retriever import DOC_Retriever
from collate_func import collateLM
import torch
import dataset
from tqdm import tqdm
import torch.utils.checkpoint
import random
from torch.utils.data import random_split
from rouge import Rouge
import numpy as np
from bert_score import score
from transformers import get_linear_schedule_with_warmup
import yaml
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda'

import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)

def calculate_rouge(a,b):
    rouge=Rouge()
    score = rouge.get_scores(a,b)
    
    scores=np.mean([score[0]["rouge-1"]['f'],score[0]["rouge-2"]['f'],score[0]["rouge-l"]['f']])
    return scores

def trainer(epoch):
    loss_mv=4.8
    acc=[]
    token_acc=[]
    rouge=[]
    E.train()
    
    bar = tqdm(train_loader, ncols=0, smoothing=0.05, desc=f'epoch:{epoch+1:03d}/{train_config["max_epoch"]}')
    
    for tokens, masks, targets, querys, answers,_ in bar:
        tokens, masks, targets = map(lambda x:x.to(device),[tokens, masks, targets])
        # print(tokens, masks, targets, querys)

        OptEnc.zero_grad()
        topkseg=retriever.retrieve(querys, k=train_config['topk'])# shape:(B, k, n)
        topkseg=topkseg.reshape(-1,topkseg.shape[-1])# shape(B*k, n)
        topkseg=topkseg.to(device)
        x={'input_ids':topkseg,'attention_mask': torch.ones_like(topkseg).to(device)}
        
        pooler, embs, embsmasks=E(x=x, k=train_config['topk'], dtype=torch.float16)
        embs.retain_grad()
        # embs.register_hook(lambda grad: grad)
        out, loss = model.forward(ids=tokens, target=targets, masks=masks, encoder_output=embs, encoder_masks=embsmasks)
        
        mask_indices =targets != -100# shape (2,24)
        
        for b in range(out.shape[0]):
            a=torch.masked_select(torch.argmax(out[b,:-1,:],dim=-1),mask_indices[b,1:])
            tar=torch.masked_select(targets[b],mask_indices[b])

            # predict=model.tokenizer.decode(a,skip_special_tokens=True)
            
            # accc=int(predict==answers[b])
            
            token_accc= (a==tar).float().mean()
            token_acc.append(token_accc)
            accc=int(token_accc)
            acc.append(accc)
            # if len(predict)!=0:
            #     rouge_score=calculate_rouge(predict,answers[b])

            #     rouge.append(rouge_score)

            
        (loss).backward()

        OptEnc.step()
        # lr_scher.step()
        loss_mv = loss_mv*0.98+loss.item()*0.02
        
        bar.set_postfix_str(f'loss:{loss_mv:.3f}, emb:{embs.grad.norm():.3f}, acc:{sum(acc)/len(acc):.4f}, tokenacc:{sum(token_acc)/len(token_acc):.4f} ')
        # if count%200==0:
        #     with torch.no_grad():
        #         pooler, embs, embsmasks=E.forward('Fernie Alpine Resort', k=1, dtype=torch.float16)
        #         if count==0:
        #             embs, embsmasks=None,None
        #         model.generate(collate_fn.template('where did they film hot tub time machine'), embs, embsmasks, 128, test=False)
        

@torch.inference_mode()
def valid():
    
    bar = tqdm(val_loader, ncols=0, smoothing=0.05, desc=f'Validation')
    acc=[]
    bestsore=[]
    E.eval()
    with torch.no_grad():
        for tokens, masks, targets, querys, answers,all_as in bar:
            
            topkseg=retriever.retrieve(querys, k=train_config['topk'])
            topkseg=topkseg.reshape(-1,topkseg.shape[-1])
            topkseg=topkseg.to(device)
            x={'input_ids':topkseg,'attention_mask': torch.ones_like(topkseg).to(device)}
            
            _, embs, embsmasks=E(x=x, k=train_config['topk'], dtype=torch.float16)
            # print(embs[:,:,0,:,:,:].unsqueeze(2).shape, embsmasks[0].unsqueeze(0).shape)
            for i in range(len(querys)):
                predict=model.generate(collate_fn.template(querys[i]), embs[:,:,i,:,:,:].unsqueeze(2), embsmasks[i].unsqueeze(0), 16, test=False,streamer=False)
                predict=predict.strip()
                found = predict in all_as[i]
                # found =predict==answers[i]
                # print('-----------------------')
                # print(querys[i])
                # print(predict)
                # print(all_as[i])
                # print(found)
                # print('-----------------------')
                

                if found:
                    accc=1
                else: 
                    accc=0

                acc.append(accc)
                P, R, F1 = score([predict], [answers[i]],lang='en', model_type='bert-base-uncased')
                bestsore.append(F1.item())
            
            bar.set_postfix_str(f'acc: {sum(acc)/len(acc):.4f} bertscore: {sum(bestsore)/len(bestsore):.4f}')
        
@torch.inference_mode()
def baseline():
    
    bar = tqdm(val_loader, ncols=0, smoothing=0.05, desc=f'Validation')
    acc=[]
    bestsore=[]
    with torch.no_grad():
        for tokens, masks, targets, querys, answers,all_as  in bar:
            for i in range(len(querys)):
                predict=model.generate(collate_fn.template(querys[i]),encoder_output = None, encoder_masks=None, max_new_tokens =5, test=False,streamer=False)
                predict=predict.replace('.', '').replace('\n', '').replace(':', '').strip()
                found = predict  in all_as[i]
                # found =predict==answers[i]

                # print('-----------------------')
                # print(querys[i])
                # print(predict)
                # print(all_as[i])
                # print(found)
                # print('-----------------------')
                
                if found:
                    accc=1
                else: 
                    accc=0
                acc.append(accc)
                P, R, F1 = score([predict], [answers[i]],lang='en', model_type='bert-base-uncased')
                bestsore.append(F1.item())
            bar.set_postfix_str(f'acc: {sum(acc)/len(acc):.4f} bertscore: {sum(bestsore)/len(bestsore):.4f}')

if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['loader_config']
   
    retriever = DOC_Retriever()
    model=LLaMa_reader("meta-llama/Llama-2-7b-chat-hf")

    num_heads = model.model.config.num_key_value_heads
    num_layers = model.model.config.num_hidden_layers
    num_dims = model.model.config.hidden_size//num_heads

    E=KnowEncoder(num_layers, num_heads, num_dims, train_config['head'])
    E.to(device)
    E.model.requires_grad_(False)
    model.tokenizer.padding_side='left'

    if train_config['load_E']:
        E.load_state_dict(torch.load('/home/devil/workspace/nlg_progress/backend/app/save/knowled_encoder.pt'))
    collate_fn = collateLM(tokenizer=model.tokenizer)
    qadataset = list(dataset.QAPairDataset(num_samples=train_config['num_samples']))
    train_size = int(train_config['spilt'] * len(qadataset))
    val_size = len(qadataset) - train_size

    train_dataset, val_dataset = random_split(qadataset, [train_size, val_size])
    
    OptEnc = torch.optim.AdamW(E.parameters(), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    # lr_scher=get_linear_schedule_with_warmup(optimizer=OptEnc,num_warmup_steps=0,num_training_steps=(len(train_dataset)*train_config['max_epoch']))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_config, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, **loader_config, collate_fn=collate_fn)

    # for i in range(train_config['max_epoch']):
    #     trainer(i)
    #     torch.save(E.state_dict(),'/home/devil/workspace/nlg_progress/backend/app/save/knowled_encoder.pt')
        
    # valid()

    baseline()