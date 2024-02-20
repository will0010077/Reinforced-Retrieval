import torch
from torch.nn.utils.rnn import pad_sequence


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
