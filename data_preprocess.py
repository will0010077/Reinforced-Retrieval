import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import torch
import json
from tqdm import tqdm
from DatasetLoader import dataset, collate_func

from RL.utils import generate_segments
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import tensor_retuen_type


if __name__=="__main__":
    process=["dataclean", "process_QAD", "", ""]
    runall=False
    device=0
    if len(sys.argv) < 2:
        print(f"You can run \"all\", or select from following, mutilple value split by \',\':")
        for i, p in enumerate(process):
            print(i,") ",p, sep="")
        s = input(">")
        if "all" in s:
            runall = True
        else:
            s= s.split(",")
            s = [int(i) for i in s]
        
        data_select = input("dataset:\n0)Natural Question\n1)Trivia\n>")
        data_select=int(data_select)
        data_name = ["NQ", "TV"]
        assert data_select==0 or data_select==1
    if runall or 0 in s:
        
        if data_select==0:
            data_path='data/v1.0-simplified_simplified-nq-train.jsonl'
            data = dataset.cleanDataset(data_path=data_path,num_samples=None)
        elif data_select==1:
            data = dataset.trivia_qadatast("train")
        dataset.cleandata(data, f"data/{data_name[data_select]}_train.jsonl")
        
        if data_select==0:
            data_path = 'data/v1.0-simplified_nq-dev-all.jsonl'
            data = dataset.cleanDataset(data_path=data_path,num_samples=None)
        elif data_select==1:
            data = dataset.trivia_qadatast("valid")
        dataset.cleandata(data, f"data/{data_name[data_select]}_test.jsonl")


    if runall or 1 in s:
        
            
        print('Loading dataset...')
        data_path = f"data/{data_name[data_select]}_train.jsonl"
        train_data = dataset.NQADataset(data_path=data_path,  num_samples=None, use_long=True, use_short=True, use_doc=True)
        collate = collate_func.collate()
        
        
        print('Initilize retriever')
        lex_MAE_retriver=lex_retriever()
        lex_MAE_retriver.to(device)
        lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)
        
        file = open(f"data/{data_name[data_select]}_train_QAD.jsonl", "w")
        for i, (q, la, a, d) in tqdm(enumerate(train_data), total=len(train_data), ncols=0, smoothing=0.05):
            d = generate_segments(d, 96, 64)
            input_ids = []
            attention_mask = []
            embedding = []
            tokens = lex_MAE_retriver.tokenizer(d, padding = True, truncation=True, max_length=256, return_tensors="pt", add_special_tokens=False)
            bs=64
            for i in range(0,len(d), bs):
                batch_tokens = tensor_retuen_type(input_ids = tokens.input_ids[i:i+bs], attention_mask = tokens.attention_mask[i:i+bs])
                batch_tokens = batch_tokens.to(lex_MAE_retriver.device)
                with torch.no_grad():
                    embedding.append(lex_MAE_retriver.forward(batch_tokens)) #(N,d)
            embedding = torch.cat(embedding)
            
            query = lex_MAE_retriver.tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(lex_MAE_retriver.device)
            with torch.no_grad():
                query = lex_MAE_retriver.forward(query)#(b,d)
            topk = torch.topk(query @ embedding.T, k = min(10, len(embedding)), dim=-1)#(1,k)->(k)
            idx = topk.indices[0].tolist()
            val = topk.values[0].tolist()
            d = [d[k] for i, k in enumerate(idx)]
            json.dump(dict(question=q, short_answers=a, long_answer=la, document = d, score = val), file)
            file.write('\n')
        file.close()
    if runall or 2 in s:
        pass
    if runall or 3 in s:
        pass
