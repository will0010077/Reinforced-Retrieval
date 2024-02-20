import sys
# sys.path.append("../../")


from tqdm import tqdm
import torch
import json
import torch.nn as nn
import random
from torch.utils.data import random_split

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


# from contriver import Contriever
seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Contriever(torch.nn.Module):
    def __init__(self):
        super(Contriever, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    def forward(self, x):
        # x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to('cuda')
        y=self.model(**x)
        y=self.mean_pooling( y[0], x['attention_mask'])
        return y
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

def vicreg(z_a,z_b):
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


    dis=z_a[:,None,:] - z_b[None,:,:]#(B,B,k)
    dis = torch.sum(dis**2, dim=2)
    ratio = dis.diag() / dis.mean(dim=1)
    ratio = ratio.mean().item()
    return loss, ratio

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
    s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
    s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity
    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N),                           # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss

    dis=u[:,None,:] - v[None,:,:]#(B,B,k)
    dis = torch.sum(dis**2, dim=2)
    ratio = dis.diag() / dis.mean(dim=1)
    ratio = ratio.mean().item()
    return loss,ratio

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
        la=self.data[idx]['long_answer']
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
        
        # a=sample['annotations'][0]['long_answer']#sample['long_answer_candidates'][random_number]
        
        # long_answer=' '.join(sample['document_text'].split()[a['start_token']:a['end_token']])
        
        return q,la#str(sample['question_text']),long_answer#text_without_tags


class collect_fn():
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    def __call__(self,batch):
        '''
            input: question,answer
            return:torch.tensor(token_id) batchxseq_len
        '''
        input_list_a=[]
        input_list_b=[]

        for Q,A in batch:
            if len(A)>0:
                input_list_a.append(Q)
                input_list_b.append(A)

        output_a=self.tokenizer (text=input_list_a,return_tensors="pt",padding=True,truncation=True,max_length=128)
        output_b=self.tokenizer (text=input_list_b,return_tensors="pt",padding=True,truncation=True,max_length=512)

        return output_a, output_b


def trainer(epoch,model):
    model.train()
    loss_ls=[]
    ratio_ls=[]
    acc=[]
    for q,a in (bar:=tqdm(train_dataloader,ncols=0)):

        optimizer.zero_grad()
        q=q.to(device)
        a=a.to(device)
    
        q = model(q)
        a = model(a)
        
        loss, ratio =xt_xent(q,a)#vicreg(q,a)
        _,ids=torch.topk(torch.mean((q[:,None,:]-a[None,:,:])**2,2),min(5,len(q)),largest=False)
        eye=torch.arange(len(q))[:,None].to(device)
        
        acc.append((eye==ids).sum()/len(ids))
        loss.backward()
        optimizer.step()
        loss_ls.append(loss)
        ratio_ls.append(ratio)
        bar.set_description(f'epoch[{epoch+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {sum(loss_ls)/len(loss_ls):.4f} ratio {sum(ratio_ls)/len(ratio_ls):.2f} acc{sum(acc)/len(acc):.4f}')
    lr_scher.step(sum(loss_ls)/len(loss_ls))
        
def validation(model):
   
    model.eval()

    embedding=[]
    querys=[]
    with torch.no_grad():
        for q,a in tqdm(val_dataloader):
            q.to(device)
            a.to(device)
            z=model(q)
            zz=model(a)
            querys.append(z.to('cpu'))
            embedding.append(zz.to('cpu'))

    embedding=torch.cat(embedding)
    querys = torch.cat(querys)
    # print(embedding.shape)
    B=256
    acc=[]
    for i in range(0, len(querys), B):
        if len(querys)-i<B:
            break
        aaa,ids=torch.topk(torch.mean((querys[i:i+B,None,:]-embedding[None,:,:])**2,2),5,largest=False)
        eye=torch.arange(i,i+B)[:,None]
        
        acc.append((eye==ids).sum()/len(ids))
    print('total acc:',sum(acc)/len(acc))
    return sum(acc)/len(acc)

if __name__=='__main__':

    data_path='/home/devil/workspace/nlg_progress/backend/app/data/cleandata.pt'
    dataset=NQADataset(data_path=data_path,)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collect_fn())
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False,collate_fn=collect_fn())

    model=Contriever()
    model.to(device)

    model_path='/home/devil/workspace/nlg_progress/backend/app/save/contriever.pt'
    # model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    num_epochs=5
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=20, min_lr=1e-5)
    bestacc=0
    for i in range(num_epochs):
        trainer(i, model)
    #     acc=validation(model)
    #     if acc>bestacc:
    #         torch.save(model.state_dict(),model_path)
    #         bestacc=acc
    # print('best acc:',bestacc)
    acc=validation(model)
    print('acc:',acc)
    torch.save(model.state_dict(),model_path)
    print('saved contriever')

    
    
    