import sys
# sys.path.append("../../")


from tqdm import tqdm
import torch
import json
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def vicreg(z_a,z_b):
    '''(B,k)'''
    lam=25
    mu=25
    nu=1
    sim_loss = nn.MSELoss()(z_a, z_b)
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

class NQADataset(Dataset):
    def __init__(self, data_path='/home/devil/workspace/nlg_progress/backend/app/data/v1.0-simplified_simplified-nq-train.jsonl',num_samples=1000):
        self.data_path = data_path
        self.num_samples = num_samples
        
        # self.len = 307000
        self.data = self.load_data()
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

        sample = self.data[idx]
        
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
                
        return str(sample['question_text']),str(sample['long_answer_candidates'])

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
            input_list_a.append(Q)
            input_list_b.append(A)



        output_a=self.tokenizer (text=input_list_a,return_tensors="pt",padding=True,truncation=True,max_length=512)
        output_b=self.tokenizer (text=input_list_b,return_tensors="pt",padding=True,truncation=True,max_length=128)

        return output_a, output_b


def trainer(epoch,model):
    model.train()
    for q,a in (bar:=tqdm(train_dataloader,ncols=0)):

        optimizer.zero_grad()
        q=q.to(device)
        a=a.to(device)
    
        q = model(q)
        a = model(a)
        
        loss, ratio = vicreg(q,a)

        loss.backward()
        optimizer.step()

        bar.set_description(f'epoch[{epoch+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {loss:.4f} ratio {ratio:.2f}')
    lr_scher.step(loss)
        

if __name__=='__main__':
    dataset=NQADataset(num_samples=307000)

    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True,collate_fn=collect_fn())
    model=Contriever()
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    num_epochs=100
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, cooldown=10, min_lr=1e-6)

    for i in range(num_epochs):
        trainer(i, model)
        torch.save(model.state_dict(),'/home/devil/workspace/nlg_progress/backend/app/save/contriever.pt')