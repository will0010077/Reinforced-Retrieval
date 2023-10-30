import torch
from torch.nn.utils.rnn import pad_sequence


class collateLM():
    def __init__(self, max_len=1024, tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len

        self.bos_id=self.tokenizer.bos_token_id
        self.eos_id=self.tokenizer.eos_token_id
        self.pad_id = self.eos_id

    def __call__(self, batch):
        '''
        input:(bs,(Q,A))
        output: batched [<bos>Q<eos>A<eos>]
        '''

        tokens=[]
        masks=[]
        targets=[]
        querys=[]
        for q, a in batch:
            q_out=self.tokenizer(q, return_tensors="pt", truncation=True, max_length=self.max_p_len-2)
            a_out=self.tokenizer(a, return_tensors="pt", truncation=True, max_length=self.max_c_len-1)
            #print(p_out.input_ids.shape)#(1, len)

            # q_ids =torch.cat([torch.tensor([self.bos_id]),q_out['input_ids'][0],torch.tensor([self.eos_id])])
            q_ids = q_out['input_ids'][0]
            if q_ids[-1] != self.eos_id:
                q_ids[-1] = self.eos_id
            p_mask=torch.ones_like(q_ids)

            a_ids =torch.cat([a_out['input_ids'][0],torch.tensor([self.eos_id])])
            a_ids = a_out['input_ids'][0]
            if a_ids[-1] != self.eos_id:
                a_ids[-1] = self.eos_id
            c_mask=torch.ones_like(a_ids)

            ids=torch.cat([q_ids,a_ids])

            tokens.append(ids)
            masks.append(torch.cat([p_mask,c_mask]))
            targets.append(torch.cat([torch.ones_like(q_ids)*-100, a_ids]))
            querys.append(a)

        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)
        targets=pad_sequence(targets, batch_first=True, padding_value=-100)
        # print( tokens.shape, masks.shape, targets.shape)#check OK
        return tokens, masks, targets, querys
