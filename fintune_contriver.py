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
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


# from contriver import Contriever
seed = config['seed']
# torch.manual_seed(seed)
# random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class NQADataset(Dataset):
    def __init__(self, data_path='/home/contriever/v1.0-simplified_simplified-nq-train.jsonl',num_samples=None):
        self.data_path = data_path

        self.num_samples = num_samples
        self.data = torch.load(self.data_path)#self.load_data()

    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break

                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        q=self.data[idx]['question']
        la=re.sub('<[/a-zA-Z0-9]*>', '',string=self.data[idx]['long_answer'])
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])

        # a=sample['annotations'][0]['long_answer']#sample['long_answer_candidates'][random_number]

        # long_answer=' '.join(sample['document_text'].split()[a['start_token']:a['end_token']])

        return q,la#str(sample['question_text']),long_answer#text_without_tags


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
    output_b = tokenizer (text=input_list_b, return_tensors="pt",padding=True,truncation=True)
    output_b = torch.cat([torch.ones([len(output_b),1], dtype=torch.long)*tokenizer.cls_token_id, output_b, torch.ones([len(output_b),1], dtype=torch.long)*tokenizer.eos_token_id], dim=1)#(B,256)
    return output_a, output_b


def trainer(epoch, model:nn.Module, early_stop=None):
    model.train()
    ma_loss=3.4
    ma_dis=0
    ma_acc=0.5
    F_lambda=0.01
    flop_loss=FLOPS()
    count=0
    for q,a in (bar:=tqdm(train_dataloader,ncols=0)):

        count+=1
        if count==early_stop:
            break
        optimizer.zero_grad()
        q=q.to(device)
        a=a.to(device)

        q = model(q)
        a = model(a)
        loss, dis =xt_xent(q, a, temperature=0.01)
        dis = dis.item()
        loss += F_lambda*(flop_loss.forward(q)+flop_loss.forward(a))#vicreg(q,a)
        # loss += -dis*0.1
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _,ids=torch.topk((q@a.T).to_dense(),5,largest=True)#(N,5)
            eye=torch.arange(len(q))[:,None].to(device)

        acc=(eye==ids).sum()/len(ids)
        ma_acc = 0.99*ma_acc + 0.01*acc
        ma_loss = 0.99*ma_loss + 0.01*loss
        ma_dis = 0.99*ma_dis + 0.01*dis
        bar.set_description(f'epoch[{epoch+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss: {ma_loss:.4f}, dis: {ma_dis:.2f}, acc: {ma_acc:.4f}')
    lr_scher.step(ma_loss)

    return ma_loss
def validation(model):

    model.eval()

    embedding=[]
    querys=[]
    with torch.no_grad():
        for q,a in tqdm(val_dataloader,ncols=0):
            q.to(device)
            a.to(device)
            z_q = model(q)
            z_a = model(a)
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
    lex_MAE_retriver.to(device)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collect_fn, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=collect_fn)


    checkpoint_path = 'save/LEX_MAE_retriever.pt'
    load_path = 'save/LEX_MAE_retriever_loss_6.1098.pt'
    if os.path.isfile(load_path):
        lex_MAE_retriver.model.load_state_dict(torch.load(load_path, map_location='cpu')['enc_model_state_dict'])
        print('load weight from',load_path)
    else:
        print('Train from scrach')

    optimizer = torch.optim.AdamW(lex_MAE_retriver.parameters(), lr=1e-5, weight_decay=1e-2)
    num_epochs=40
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=20, min_lr=1e-5)
    bestacc=0.80
    for i in range(num_epochs):
        loss=trainer(i, lex_MAE_retriver, early_stop=2000)
        acc=validation(lex_MAE_retriver)
        if acc>bestacc:
            torch.save({'enc_model_state_dict': lex_MAE_retriver.model.state_dict()}, checkpoint_path.replace(".pt",f"{int(acc*1000):03d}.pt"))
            bestacc=acc

    acc=validation(lex_MAE_retriver)
    print('acc:',acc)

    print('saved contriever')



