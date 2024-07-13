import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
from DocBuilder.utils import tensor_retuen_type
from config import LM_dir, bert_dir
class collateLM():
    def __init__(self, max_len=1024, tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer
        self.max_len=max_len
        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len

        self.bos_id=self.tokenizer.bos_token_id
        self.eos_id=self.tokenizer.eos_token_id
        self.pad_id = self.eos_id
    def template(self, q):
        return 'The answer of the question: '+q+' is'

    def __call__(self, batch):
        '''
        input:(bs,(Q,A))
        output: batched [<bos>Q<eos>A<eos>]
        '''

        tokens=[]
        masks=[]
        targets=[]
        querys=[]
        answers=[]
        all_as=[]
        for q, a ,all_a in batch:
            out=self.tokenizer(self.template(q)+' '+a+self.tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=self.max_len)
            q_out=self.tokenizer(self.template(q), return_tensors="pt", truncation=True, max_length=self.max_len)
            ids = out['input_ids'][0]
            mask = out['attention_mask'][0]
            target = ids.clone()
            target[:q_out['input_ids'].shape[1]] = -100

            tokens.append(ids)
            masks.append(mask)
            targets.append(target)
            querys.append(q)
            answers.append(a)
            all_as.append(all_a)
        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)
        targets=pad_sequence(targets, batch_first=True, padding_value=-100)
        # print( tokens.shape, masks.shape, targets.shape)#check OK
        return tokens, masks, targets, querys, answers,all_as



class collate():
    def __init__(self, LM_dir = LM_dir, bert_dir = bert_dir):
        
        self.datatokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.LMtokenizer = AutoTokenizer.from_pretrained(
            LM_dir, use_fast=True, lstrip=False, 
            token='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')
        self.LMtokenizer.pad_token = self.LMtokenizer.eos_token
        self.eos_token = self.LMtokenizer.eos_token
    def prepare_QA_token(self, texts:list[str], targets:list[str]):
        
        unlabel_str, label = zip(*[self.templete(q, a) for q,a in zip(texts, targets)])
        cat_qa = [q+" "+a+self.eos_token for q, a in zip(unlabel_str, label)]
        unlabel = self.LMtokenizer(text=unlabel_str).input_ids
        # print(max([len(s) for s in unlabel]))
        tokens = self.LMtokenizer(text=cat_qa, text_target = cat_qa,  return_tensors='pt', padding=True, max_length=512, truncation =True,)
        
        for i in range(len(texts)):
            tokens['labels'][i, :len(unlabel[i])]=-100
        tokens['labels'][tokens['attention_mask']==0]=-100
        return unlabel, unlabel_str, tokens

    def collate_qa(self, batch:list):
        q_str, a_str = [*zip(*batch)]
        q_str, a_str = list(q_str), list(a_str)
        unlabel, unlabel_str, qa_tokens = self.prepare_QA_token(q_str, a_str)
        a_tokens = self.datatokenizer(a_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        return tensor_retuen_type(**qa_tokens), unlabel, unlabel_str, q_str, a_str, tensor_retuen_type(**a_tokens)
    
    def collate_q(self, batch:list):
        batch = [self.templete(q, a) for q, a in batch]
        q_str, a_str = [*zip(*batch)]
        q_str, a_str = list(q_str), list(a_str)
        tokens = self.LMtokenizer(text=q_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        a_tokens = self.datatokenizer(a_str, return_tensors='pt', padding=True, max_length=256, truncation =True,)
        return tensor_retuen_type(**tokens), q_str, a_str, tensor_retuen_type(**a_tokens)
    
    def templete(self, query:str, answer:str ='')->tuple[str]:
        Role = ["system", "user", "assistant"]
        query, answer = query.strip(), answer.strip()
        messages = [
            {"role": "system", "content": "This is the searched knowledge: [KNOW]  [/KNOW] Please answer user questions based on the above knowledge\n"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
        prompt = self.LMtokenizer.apply_chat_template(
            messages[:2],
            tokenize=False, 
            add_generation_prompt=True,
            return_tensors="pt"
        )
        return prompt, answer
    
    def state_templete(self, q, a, action):
        sep = self.datatokenizer.sep_token
        return " ".join(action)+ sep+ q+ sep+ a