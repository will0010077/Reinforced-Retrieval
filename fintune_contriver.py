import sys
# sys.path.append("../../")


from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
import json
import random
import re
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
import yaml,os
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import top_k_sparse, tensor_retuen_type
from DatasetLoader.dataset import NQADataset
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# from contriver import Contriever
seed = config['seed']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FLOPS(nn.Module):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def vicreg(z_a:Tensor, z_b:Tensor):
    '''(B,k)'''
    lam=1
    mu=1
    nu=0.4
    sim_loss = nn.MSELoss()(z_a, z_b)
    #dot_prodot=torch.mean((z_a@z_b.T).diag())
    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    # covariance loss
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    N=z_a.shape[0]
    D=z_a.shape[1]
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = torch.masked_select(cov_z_a, (1-torch.eye(cov_z_a.shape[0],device=device)).to(torch.bool)).pow_(2).sum() / D
    + torch.masked_select(cov_z_b, (1-torch.eye(cov_z_b.shape[0],device=device)).to(torch.bool)).pow_(2).sum() / D
    # loss
    loss = lam * sim_loss + mu * std_loss + nu * cov_loss


    dis = z_a@z_b.T
    dis = dis.diag() - dis.mean(dim=1)
    dis = dis.mean().item()
    return loss, dis

def xt_xent(
    u: torch.Tensor,                               # [N, C]
    v: torch.Tensor,                               # [N, C]
    temperature: float = 0.1,
):
    """
    N: batch size
    C: feature dimension
    """
    N, C = u.shape

    z = torch.cat([u, v], dim=0)                   # [2N, C]
    z = F.normalize(z, p=2, dim=1)                 # [2N, C]
    s = torch.matmul(z, z.T) / temperature       # [2N, 2N] similarity matrix

    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix

    s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity

    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N),                           # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss

    dis = u@v.T
    dis = dis.diag() - dis.mean(dim=1)
    dis = dis.mean()
    return loss, dis

tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
def collect_fn(batch):
    '''
        input: question,answer
        return:torch.tensor(token_id) batch, seq_len
    '''
    input_list_a=[]
    input_list_b=[]

    for Q,A in batch:
        if len(A)>0:
            input_list_a.append(Q)
            input_list_b.append(A)

    output_a = tokenizer (text=input_list_a, return_tensors="pt",padding=True,truncation=True, max_length=128)
    output_b = tokenizer (text=input_list_b, return_tensors="pt",padding=True,truncation=True, max_length=128)
    # output_b = torch.cat([torch.ones([len(output_b),1], dtype=torch.long)*tokenizer.cls_token_id, output_b, torch.ones([len(output_b),1], dtype=torch.long)*tokenizer.eos_token_id], dim=1)#(B,256)
    return tensor_retuen_type(**output_a), tensor_retuen_type(**output_b)

F_lambda=2**5
def trainer(epoch, model:nn.Module, early_stop=None):
    global F_lambda
    model.train()
    ma_loss=2
    ma_dis=0
    ma_acc=0.5
    flop_loss=FLOPS()
    count=0
    stream = torch.cuda.current_stream()
    
    for q,a in (bar:=tqdm(train_dataloader,ncols=0)):

        count+=1
        if count==early_stop:
            break
        optimizer.zero_grad()
        q=q.to(device)
        a=a.to(device)
        q = model(q)
        a = model(a)
        q = F.normalize(q, dim=-1)
        a = F.normalize(a, dim=-1)
        loss, dis =xt_xent(q, a, temperature=1/2**7)
        dis = dis.item()
        loss += F_lambda*(flop_loss.forward(q)+flop_loss.forward(a))#vicreg(q,a)
        # loss += -dis*0.1
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _,ids=torch.topk((q@a.T).to_dense(),5,largest=True)#(N,5)
            eye=torch.arange(len(q))[:,None].to(device)
            sparse_error = 0.5*torch.norm(q - top_k_sparse(q.detach(), config['cluster_config']['k_sparse']).to_dense(), dim=-1).mean()
            sparse_error += 0.5*torch.norm(a - top_k_sparse(a.detach(), config['cluster_config']['k_sparse']).to_dense(), dim=-1).mean()

        acc=(eye==ids).sum()/len(ids)
        ma_acc = 0.99*ma_acc + 0.01*acc
        ma_loss = 0.99*ma_loss + 0.01*loss
        ma_dis = 0.99*ma_dis + 0.01*dis
        bar.set_description(f'epoch[{epoch+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss: {ma_loss:.4f}, dis: {ma_dis:.2f}, acc: {ma_acc:.4f}, err:{sparse_error:.4f}')
        if count>1000 and ma_acc<0.2:
            model.model.load_state_dict(torch.load(load_path, map_location='cpu')['enc_model_state_dict'])
            F_lambda = F_lambda/1.5
            print('lambda',F_lambda)
            return trainer(epoch, model, early_stop)

    return ma_loss
def validation(model:nn.Module):

    model.eval()

    embedding=[]
    querys=[]
    with torch.no_grad():
        for q,a in tqdm(val_dataloader,ncols=0):
            q.to(device)
            a.to(device)
            z_q = model(q)
            z_a = model(a)
            z_q = F.normalize(z_q, dim=-1)
            z_a = F.normalize(z_a, dim=-1)
            z_q = top_k_sparse(z_q, config['cluster_config']['k_sparse']).to_dense()
            z_a = top_k_sparse(z_a, config['cluster_config']['k_sparse']).to_dense()
            querys.append(z_q.to('cpu'))
            embedding.append(z_a.to('cpu'))

    embedding=torch.cat(embedding)
    querys = torch.cat(querys)
    # print(embedding.shape)
    B=256
    acc=[]
    # for i in tqdm(range(0, len(querys), B)):
    count=0
    _,ids=torch.topk((querys@embedding.T).to_dense(),5,largest=True)#(N,5)
    eye=torch.arange(querys.shape[0])[:,None]

    acc.append((eye==ids).sum()/len(ids))
    print('total acc:',sum(acc)/len(acc))
    return sum(acc)/len(acc)


if __name__=='__main__':

    data_path='data/cleandata.pt'
    # print(torch.load(data_path))
    dataset=NQADataset(data_path=data_path)
    print(len(dataset))
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size


    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.train()

    torch.manual_seed(seed)
    random.seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=192, shuffle=True,collate_fn=collect_fn, num_workers=6, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=collect_fn)


    checkpoint_path = 'save/LEX_MAE_retriever.pt'
    load_path = 'save/LEX_MAE_retriever_loss_6.8032.pt'
    if os.path.isfile(load_path):
        lex_MAE_retriver.model.load_state_dict(torch.load(load_path, map_location='cpu')['enc_model_state_dict'])
        print('load weight from',load_path)
    else:
        print('Train from scrach')
    
    lex_MAE_retriver.to(device)
    lex_MAE_retriver = nn.DataParallel(lex_MAE_retriver, device_ids=[0,1])
    print(lex_MAE_retriver)
    optimizer = torch.optim.AdamW(lex_MAE_retriver.parameters(), 
                lr=config['lex']['fine_lr'])
    num_epochs=40
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, cooldown=3, min_lr=config['lex']['fine_lr']/10)
    bestacc=0.88
    for i in range(num_epochs):
        loss=trainer(i, lex_MAE_retriver, early_stop=2000)
        lr_scher.step(loss)
        acc=validation(lex_MAE_retriver)
        if acc>bestacc:
            torch.save({'enc_model_state_dict': lex_MAE_retriver.model.state_dict()}, checkpoint_path.replace(".pt",f"{int(acc*1000):03d}.pt"))
            bestacc=acc

    acc=validation(lex_MAE_retriver)
    print('acc:',acc)

    print('saved contriever')



