import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file, load_file
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.LexMAE import lex_retriever


from DocBuilder.utils import tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder
from DatasetLoader.dataset import NQADataset
from metric.reward import BLEU_score, Bert_score,ROUGE_score
from RL.utils import generate_segments
from tqdm import tqdm
import yaml
import peft
import os

from PrefixPretrain import collate
import config
token = "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"
model_dir = "meta-llama/Llama-2-7b-chat-hf"


class small_retriever:
    def __init__(self, doc, model:lex_retriever):
        self.len = len(doc)
        self.document = [*doc]
        self.input_ids = [None]*self.len
        self.attention_mask = [None]*self.len
        self.embedding = [None]*self.len
        self.ret = model
        self.collate=collate()
        for idx in range(self.len):
            self._build_embedding(idx)
    @torch.no_grad()
    def _build_embedding(self, idx):
        self.document[idx] = generate_segments(self.document[idx],96,64)[:256]
        tokens = self.collate.datatokenizer(self.document[idx], padding = True, truncation=True, max_length=256, return_tensors="pt", add_special_tokens=False).to(self.ret.device)
        self.input_ids[idx] = tokens.input_ids
        self.attention_mask[idx] = tokens.attention_mask
        self.embedding[idx] = self.ret.forward(tokens)#(N,d)

    @torch.no_grad()
    def retrieve(self, ids:list, x:list[str]):
        query = self.ret.tokenizer(x, return_tensors="pt", padding=True).to(self.ret.device)
        query = self.ret.forward(query)#(b,d)
        retrieved = []
        for idx, q in zip(ids, query):
            topk = torch.argmax(q[None] @ self.embedding[idx].T, dim=-1)[0]#(1,1)->()
            retrieved.append(self.input_ids[idx][topk][:sum(self.attention_mask[idx][topk])])
        return retrieved
    
if __name__=="__main__":
    device = 0
    print('Loading LLM')
    LM = LLaMa_reader(model_dir, 'cpu', token = token, from_pretrained=True)
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

    print('Initilize retriever')
    
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=True)
    lex_MAE_retriver.to(device)
    
    
    max_epoch = 1
    print('Loading dataset...')
    data_path = "data/dev_with_doc.jsonl"
    dataset = NQADataset(data_path=data_path, num_samples=80, use_doc=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, persistent_workers=True)

    ori_true_bert_list = []
    ori_ret_bert_list = []
    pre_true_bert_list = []
    pre_ret_bert_list = []
    
    ori_true_bleu_list = []
    ori_ret_bleu_list = []
    pre_true_bleu_list = []
    pre_ret_bleu_list = []
    f = open("moniter.txt", "a")
    for i, (q_str, a_str, doc) in enumerate(loader):
        q_str, a_str, doc = [*q_str], [*a_str], [*doc]
        ret = small_retriever(doc, lex_MAE_retriver)
        
        
        d_t = ret.retrieve(range(ret.len), q_str)
        qa_tokens, unlabel, unlabel_str, q_str, a_str, a_tokens = collate().collate_qa(zip(q_str, a_str))
        a_tokens = a_tokens.to(LM.device)
        documents = collate().datatokenizer.batch_decode(d_t)
        d_t = collate().datatokenizer(documents, return_tensors="pt", padding=True).to(LM.device)
        
        with torch.no_grad():
            message = [unlabel_str[j] for j in range(len(unlabel_str))]
            output_pre_true = LM.generate(message, Doc_tokens=a_tokens, max_new_tokens=256)
            
            message = [unlabel_str[j] for j in range(len(unlabel_str))]
            output_pre_ret = LM.generate(message, Doc_tokens=d_t, max_new_tokens=256)
            
            message = [unlabel_str[j][:57]+ a_str[j]+ unlabel_str[j][57:] for j in range(len(unlabel_str))]
            output_ori_true = LM.generate(message, Doc_tokens=None, max_new_tokens=256)

            message = [unlabel_str[j][:57]+ documents[j]+ unlabel_str[j][57:] for j in range(len(unlabel_str))]
            output_ori_ret = LM.generate(message, Doc_tokens=None, max_new_tokens=256)

        # output_ori = [ output_ori[j][len(q_str[j]):] for j in range(len(q_str))]
        # output_pre = [ output_pre[j][len(q_str[j]):] for j in range(len(q_str))]

        ori_bert_true=Bert_score(output_ori_true, list(a_str))
        ori_bert_ret=Bert_score(output_ori_ret, list(a_str))
        pre_bert_true=Bert_score(output_pre_true, list(a_str))
        pre_bert_ret=Bert_score(output_pre_ret, list(a_str))

        ori_bleu_true=ROUGE_score(output_ori_true, list(a_str))
        ori_bleu_ret=ROUGE_score(output_ori_ret, list(a_str))
        pre_bleu_true=ROUGE_score(output_pre_true, list(a_str))
        pre_bleu_ret=ROUGE_score(output_pre_ret, list(a_str))

        ori_true_bert_list+=ori_bert_true
        ori_ret_bert_list+=ori_bert_ret
        pre_true_bert_list+=pre_bert_true
        pre_ret_bert_list+=pre_bert_ret

        ori_true_bleu_list+=ori_bleu_true
        ori_ret_bleu_list+=ori_bleu_ret
        pre_true_bleu_list+=pre_bleu_true
        pre_ret_bleu_list+=pre_bleu_ret

        for j in range(len(q_str)):
            f.write(
f'''Prompt: {message[j]}\nGround truth: {a_str[j]}
[{ori_bert_true[j]:.3f}, {ori_bleu_true[j]:.3f}] Original true response: {output_ori_true[j]}
[{ori_bert_ret[j]:.3f}, {ori_bleu_ret[j]:.3f}] Original ret response: {output_ori_ret[j]}
[{pre_bert_true[j]:.3f}, {pre_bleu_true[j]:.3f}] Prifix true response: {output_pre_true[j]}
[{pre_bert_ret[j]:.3f}, {pre_bleu_ret[j]:.3f}] Prifix ret response: {output_pre_ret[j]}
''' +"="*80+"\n")
        
        
        
    f.write(f"Original true bert: {sum(ori_true_bert_list)/len(ori_true_bert_list)}\n")
    f.write(f"Original ret bert: {sum(ori_ret_bert_list)/len(ori_ret_bert_list)}\n")
    f.write(f"prefix true bert: {sum(pre_true_bert_list)/len(pre_true_bert_list)}\n")
    f.write(f"prefix ret bert: {sum(pre_ret_bert_list)/len(pre_ret_bert_list)}\n")
    
    f.write(f"Original true ROUGE_score: {sum(ori_true_bleu_list)/len(ori_true_bleu_list)}\n")
    f.write(f"Original ret ROUGE_score: {sum(ori_ret_bleu_list)/len(ori_ret_bleu_list)}\n")
    f.write(f"prefix true ROUGE_score: {sum(pre_true_bleu_list)/len(pre_true_bleu_list)}\n")
    f.write(f"prefix ret ROUGE_score: {sum(pre_ret_bleu_list)/len(pre_ret_bleu_list)}\n")