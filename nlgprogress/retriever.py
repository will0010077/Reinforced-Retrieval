import openai
import json
import logging

from app.lib import mongodb

import torch
import torch.nn as nn

from transformers import AutoModel ,AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])


def check_Qmark(text:str):
    # Reduce sensitivity to question marks
    text=text.replace('？','?')
    while '??' in text:
        text=text.replace('??','?')
    if '?' not in text:
        text+='?'
    return text


class Retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model=SBert()
        self.model.eval()

    @torch.inference_mode()
    def get_feature(self, texts)->torch.Tensor:
        '''
        texts: text list with shape:(N)\\
        return: tensor with shape:(N, 768)
        '''
        feature_list=[]
        text_list=[]

        dataloader = DataLoader(texts, batch_size=32, shuffle=False)
        for texts in (bar:=tqdm(dataloader,ncols=0)):
            bs=len(texts)

            feature  = self.model(texts)#(bs, d)
            feature_list.append(feature)
            text_list.extend(texts)


        feature_list=torch.cat(feature_list)


        return  feature_list#, text_list

    def build_index(self, texts:list[str]):
        self.Q  =list(filter(lambda x:type(x)==str, texts))
        texts=[]
        for q in self.Q:
            text=check_Qmark(q)
            texts.append(text)
        self.feature= self.get_feature(texts)


    @torch.inference_mode()
    def retrieve(self, query:str, k=5, threshold=0.8):
        '''
        return k retrieved id and similarity
        '''

        query_feature = self.model(query)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        #cosine similarity
        sim = cos_sim(query_feature, self.feature)[0]

        #top-k vector and index
        v, id = torch.topk(sim, k, dim=0, largest=True)
        # scale to [0,100]
        scale=lambda x: max(0, min(100, x*100))
        return [{'question':self.Q[idx], 'score':scale(sim.item())} for idx, sim in zip(id[v>threshold], v[v>threshold])]


class SBert(torch.nn.Module):
    def __init__(self):
        super(SBert, self).__init__()
        self.model = SentenceTransformer("uer/sbert-base-chinese-nli")

    def forward(self, x):

        x = self.model.encode(x)
        x=torch.from_numpy(x).to(self.model.device)
        return x


def query_questions(file_path: str):
    try:
        with open(file_path, 'r') as file:
            group = []
            for line in file:
                group.append(line.strip())

        return group

    except FileNotFoundError:
        # Handle the case when the file is not found.
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        # Handle the case when the file is not found.
        logging.error(f"Error occurred: {e}")



def compare_text(text_to_compare: str):

    logging.info(f'query question: {text_to_compare}')
    candidate =R.retrieve(query = check_Qmark(text_to_compare))
    if len(candidate)>0:
        print(candidate)
        question = candidate[0]['question']
        result = mongodb.query_mongodb_question(question)
        if result is not None:
            logging.info(f'similar text: {question}')
            return  {'message' : result['answer'],
                     'score' : candidate[0]['score']}
        else:
            logging.error("No matching text found in db.")
            return {'message': f'No matching text found in db. '+question, 'score': 0}

    return {'message': 'No similar text.', 'score': 0}



file_path = 'app/data/question_list.txt'
question_array = query_questions(file_path)
#initial a retriever
R = Retriever()

# use gpu(if available)
# only use cpu is fine.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

R.to(device)
#pre-compute the vector of all questions
R.build_index(question_array)

