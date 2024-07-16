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

from tqdm import tqdm
import yaml
import peft
import os

from PrefixPretrain import collate
import config
token = "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"
model_dir = "meta-llama/Llama-2-7b-chat-hf"

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

    print('Initilize retriever')
    
    cluster_config=config.cluster_config
    cluster = cluster_builder(k=cluster_config.k)
    cluster.load('05_29_14_30')
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=True)
    data=torch.load('data/data_reduced_2000000.pt') ## shape:(N,d)
    retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
    retriever.to(device)
    retriever.model.to(device)
    del lex_MAE_retriver, data, cluster
    
    if True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/EncLM.pt", map_location='cpu'))
    max_epoch = 1
    print('Loading dataset...')
    data_path = "data/cleandata_with_doc.jsonl"
    dataset = NQADataset(data_path=data_path, num_samples=10000)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate().collate_qa, persistent_workers=True)

    ori_true_bert_list = []
    ori_ret_bert_list = []
    pre_true_bert_list = []
    pre_ret_bert_list = []
    
    ori_true_bleu_list = []
    ori_ret_bleu_list = []
    pre_true_bleu_list = []
    pre_ret_bleu_list = []
    f = open("moniter.txt", "a")
    for i,(tokens, unlabel, unlabel_str,  q_str, a_str, a_tokens) in enumerate(loader):
        
        tokens = tokens.to(device)
        del tokens['labels']
        tokens.input_ids = tokens.input_ids[:,:128]
        tokens.attention_mask = tokens.attention_mask[:,:128]
        a_tokens = a_tokens.to(device)
        d_t, z_t = retriever.retrieve(q_str, k=1, num_search=4)
        d_t = d_t.squeeze(1)
        documents = collate().datatokenizer.batch_decode(d_t)
        d_t = tensor_retuen_type(input_ids = d_t, attention_mask = torch.ones_like(d_t)).to(LM.device)
        
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
        
        # print(LM_output, list(a_str))
        if i==1:
            break
        
        
    f.write(f"Original true bert: {sum(ori_true_bert_list)/len(ori_true_bert_list)}\n")
    f.write(f"Original ret bert: {sum(ori_ret_bert_list)/len(ori_ret_bert_list)}\n")
    f.write(f"prefix true bert: {sum(pre_true_bert_list)/len(pre_true_bert_list)}\n")
    f.write(f"prefix ret bert: {sum(pre_ret_bert_list)/len(pre_ret_bert_list)}\n")
    
    f.write(f"Original true ROUGE_score: {sum(ori_true_bleu_list)/len(ori_true_bleu_list)}\n")
    f.write(f"Original ret ROUGE_score: {sum(ori_ret_bleu_list)/len(ori_ret_bleu_list)}\n")
    f.write(f"prefix true ROUGE_score: {sum(pre_true_bleu_list)/len(pre_true_bleu_list)}\n")
    f.write(f"prefix ret ROUGE_score: {sum(pre_ret_bleu_list)/len(pre_ret_bleu_list)}\n")