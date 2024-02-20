import torch
from torch import nn,Tensor
from torch.utils.data import dataloader
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer,AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from tqdm import tqdm
import yaml
import math
import random


with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
sep='</s>'
eos='</s>'
def prepare_inputs_for_generation(
        input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
    position_ids = kwargs.get("position_ids", None)
    if past_key_values is not None:
        # print(past_key_values[0][0].shape[-2], end='')
        # print(input_ids.shape[1], end=' ')
        position_ids = torch.ones([input_ids.shape[0],1], dtype=torch.long)*(input_ids.shape[1]-1)

        new_key_value=[]
        for i in range(len(past_key_values)):
            layer_key_value=[]
            for j in range(len(past_key_values[i])):
                layer_key_value.append(past_key_values[i][j].expand([input_ids.shape[0],-1,-1,-1]))#(B, 40, n*k, 128)
            new_key_value.append(torch.stack(layer_key_value))
        past_key_values = torch.stack(new_key_value)
        input_ids = input_ids[:, -1:]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
class LLaMa_reader:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token='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')
        self.tokenizer.model_max_length=2048
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.eos_id

        self.generate_config=config['generate_config']
        # self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
        #     trust_remote_code=False,
        #     device_map="auto",
        #     # use_triton=False,
        #     # use_cache=True,
        #     # trainable=True,
        #     )
        self.model=AutoModelForCausalLM.from_pretrained(model_dir, token='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', device_map='cpu', torch_dtype=torch.float16)
        # print(self.model)
        self.model.to(device)
        self.model.prepare_inputs_for_generation = prepare_inputs_for_generation
        self.model.training=True
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=None

    def forward(self, ids, target=None, masks=None, encoder_output = None, encoder_masks=None):
        '''
        forward function for teacher forcing\\
        the shape of ids, target, masks is (B,n)\\
        the shape of encoder_output is (B, 40, 2, 40, nd, 128), encoder_masks is (B,nd)\\
        '''


        #concat document mask
        if encoder_masks is not None:
            if masks is None:
                masks = torch.ones_like(ids, dtype = torch.long, device=encoder_masks.device)
            attention_mask = torch.cat([encoder_masks, masks.to(encoder_masks.device)],dim=-1)
        else:
            attention_mask = None
        position_ids = torch.arange(ids.shape[1]).tile([ids.shape[0],1])
        output = self.model(input_ids=ids,
                            position_ids = position_ids,
                            attention_mask=attention_mask,
                            labels=target,
                            past_key_values=encoder_output,
                            use_cache=True)

        return output.logits, output.loss

    @torch.inference_mode()
    def generate(self, message, encoder_output = None, encoder_masks=None, max_new_tokens = 1024, test=False, streamer=True):
        '''
        for inference, batch size is one.
        '''
        #tokenize
        if streamer:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer=None
        tokens = self.tokenizer(message, padding=True ,truncation=True,return_tensors='pt')
        ids=tokens.input_ids.to(self.model.device)
        masks=tokens.attention_mask.to(self.model.device)
        message_len=ids.shape[1]

        if test:#for testing encoder output
            print(tokens)
            B=1
            n=10
            
            num_heads = self.model.config.num_key_value_heads
            num_layers = self.model.config.num_hidden_layers
            num_dims = self.model.config.hidden_size//num_heads
            encoder_output=torch.zeros([num_layers,2,B,num_heads,n,num_dims], dtype=torch.float16).to(self.model.device)
            encoder_masks = torch.zeros([B,n], dtype=torch.long).to(self.model.device)

        #concat document mask
        if encoder_output is not None:
            attention_mask = torch.cat([encoder_masks, masks],dim=-1)
            position_ids = torch.arange(ids.shape[1]).tile([ids.shape[0],1])
            # position_ids=None

            #need forward first because generate will only use last ids
            output=self.model.forward(input_ids = ids[:,:-1],
                                        attention_mask=attention_mask[:,:-1],
                                        position_ids = position_ids[:,:-1],
                                        past_key_values = encoder_output,
                                        use_cache=True )

            past_key_values = output.past_key_values
        else:
            attention_mask = None
            past_key_values = None
            position_ids=None


        generate_ids = self.model.generate(input_ids = ids,
                                           attention_mask = attention_mask,
                                           past_key_values = past_key_values,
                                           use_cache=True,
                                           max_new_tokens=max_new_tokens,
                                           streamer=streamer,
                                           **self.generate_config)

        output = self.tokenizer.decode(generate_ids[0][ids.shape[1]:],skip_special_tokens=True)
        self.chat_history.append([message, output])
        return output


class KnowEncoder(torch.nn.Module):
    def __init__(self, num_layers=40, num_heads=40, dims=128, groups=4):
        super().__init__()
        if dims % groups !=0:
            raise ValueError(f'Dims must divided by groups')
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dims=dims
        self.encoder_heads=nn.Conv1d(config['embed_dim']*self.num_layers*2, self.num_heads*self.dims*self.num_layers*2, kernel_size=1, groups=groups*self.num_layers*2)
    
    def forward(self, x, k=0, dtype=torch.float32)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        # print(len(x))
        # if k==0:
        #     k=len(x)
        # assert len(x)%k==0
        # print(type(x))
        if type(x)==str:
            x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to(self.model.device)
        y=self.model(input_ids = x.get('input_ids',None), attention_mask = x.get('attention_mask',None))
        B=y[0].shape[0]//k
        n=y[0].shape[1]
        pooler_output=self.mean_pooling( y[0], x['attention_mask'])
        y = torch.tile(y[0],[1,1,self.num_layers*2])#(B*k,n,768*40*2)
        y = y.transpose(1,2)#(B*k,768*40,n)
        y = self.encoder_heads(y)#(B*k,5120*40,n)
        y=y.to(dtype)
        y = y.reshape([B, k, self.num_layers, 2, self.dims, self.num_heads, n])
        y = y.permute([0,1,2,3,5,6,4])#(B, k, 40, 2, 40, n, 128)

        #concat k to n
        batch=[]
        masks=[]
        for i in range(B):
            cat=[]
            for j in range(k):
                masked_feature = y[i,j,...,:x['attention_mask'][i*k+j].sum(dim=0),:]
                cat.append(masked_feature)
            cat = torch.cat(cat, dim=-2)#(40, 2, 40, n*k, 128)
            cat = cat.permute(3,0,1,2,4)#(n*k, 40, 2, 40, 128)
            batch.append(cat)
            masks.append(torch.ones([cat.shape[0]], dtype=torch.long, device=cat.device))
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True)#(B, n*k)
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)#(B, n*k, 40, 2, 40, 128)
        batch = batch.permute(2,3,0,4,1,5)#(40, 2, B, 40, n*k, 128)

        return pooler_output.to(dtype), batch, masks
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])

def MSE(a:torch.Tensor, b:torch.Tensor):
    '''a:(B,d), b:(N,d)\\
    out:(B,N)'''
    return torch.mean((a[:,None,:] - b[None,:,:])**2, dim=2)


def check_Qmark(text:str):
    # Reduce sensitivity to question marks
    text=text.replace('？','?')
    while '??' in text:
        text=text.replace('??','?')
    if '?' not in text:
        text+='?'
    return text


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000, scale=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe*scale
        self.register_buffer('pe', pe)
        self.pe:Tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = torch.cat([x, self.pe[:,:x.size(1)].tile(x.shape[0],1,1)], dim = -1)
        return self.dropout(x)
    
class perturb_model(torch.nn.Module):
    
    def __init__(self, in_dim=768, dim=768, num_heads=4, num_layers=2, dropout=0.1, pos_dim = 64):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim, dropout=dropout,batch_first=True)
        self.model=torch.nn.TransformerEncoder(self.layer, num_layers)
        # self.model = torch.nn.ModuleList([torch.nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.lam = torch.nn.Parameter(torch.tensor(-10,dtype=torch.float))
        self.pos_encoder = PositionalEncoding(pos_dim, dropout=dropout, max_len=16, scale=0.01)
        self.dim=dim
        self.in_dim=in_dim
        
        self.scale1=torch.nn.Linear(in_dim + pos_dim, dim, bias=True)
        self.scale2=torch.nn.Linear(dim, in_dim, bias=True)
        self.scale1.weight.data*=1e-2
        self.scale1.weight.data[torch.arange(in_dim),torch.arange(in_dim)]=torch.ones([in_dim])
        self.scale2.weight.data*=1e-2
        self.scale2.weight.data[torch.arange(in_dim),torch.arange(in_dim)]=torch.ones([in_dim])
        
        # for n,p in self.model.named_parameters():
        #     if 'weight' in n and len(p.data.shape)==2:
        #         for i in range(p.data.shape[0]//dim):
        #             p.data[i*dim:(i+1)*dim] = torch.eye(dim, dtype=p.data.dtype)
        #             p.data[i*dim:(i+1)*dim] += torch.randn([dim,dim], dtype=p.data.dtype)*1e-2
                    
    def forward(self, x:torch.Tensor, mask=None):
        '''
        x: (B,n,d)
        mask: (n,n)
        out: shape of x
        '''
        x = self.pos_encoder(x)
        x = self.scale1(x)
        
        if mask is None:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device)
        x=self.model.forward(x, mask)# + x #  (torch.nn.functional.sigmoid(self.lam))
            
        x = self.scale2(x)
        return x
    
    @torch.no_grad()
    def next(self, x:torch.Tensor, mask=None):
        '''
        x: (B,n,d)
        output: (B,d)
        '''
        x = self.forward(x, mask)
        
        return x[:,-1,:]
    
    
def prepare_parallel(query: Tensor, z: Tensor)->Tensor:
    '''
    query: (B,1,d)
    z: (B,k,d)
    returns: (B,2k,d)
    '''
    query = query.tile(1, z.shape[-2], 1)
    
    return torch.cat([query, z], dim=-2)
    
    
    
    
def parallel_mask(sz = 5, device = 'cpu'):
    '''
    output: shape(2 sz, 2 sz)
    '''
    #get diag=0, other=-inf
    mask1 = torch.full([sz]*2, float('-inf'), dtype=torch.float, device=device)
    mask1[torch.arange(sz), torch.arange(sz)] = 0.0
    
    #get lower tri=0, other=-inf
    mask2 = torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=torch.float, device=device),
        diagonal=0,
    )
    
    mask_top = torch.cat([mask1, mask2], dim=1)#(sz, 2sz)
    
    mask2[torch.arange(sz), torch.arange(sz)] = 0.0
    mask_low = torch.cat([mask1, mask2], dim=1)#(2sz,2sz)
    mask = torch.cat([mask_top, mask_low], dim=0)#(2sz,2sz)
    
    return mask
    
     
    
    

if __name__=='__main__':
    device='cuda'
    dim=128
    model=perturb_model(in_dim=dim, dim = 1024, num_heads=8, num_layers=4, dropout=0.0)
    model.to(device)
    
    num_sample=2**13
    reduce=2**10
    L=5
    Bs=256
    
    
    #toy sample
    #random generate sample with shape [num_sample, Length, dim]
    train_x = torch.randn([num_sample, 1, dim],device=device)
    train_y = train_x + torch.randn([num_sample, L, dim], device=device)*0.1 
    print('shape of x: ',train_x.shape)
    
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5, betas=[0.8,0.99], weight_decay=0.01)
    lr_schedu = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1, total_iters=100)
    train_loader=DataLoader(train_x, batch_size=Bs, shuffle=True)
    
    losses = 1
    num_epoch = [0, 40]
    bar = tqdm(range(num_epoch[0]), ncols=100)
    for epoch in bar:
        bar.set_description_str(f'epoch: {epoch}/{num_epoch[0]}')
        lr_schedu.step()
        for i, x in enumerate(train_loader):
            lr_schedu.step()
            y = x + torch.randn([x.shape[0], L, dim], device=device)*0.1
            x_ = torch.cat([x,y],dim = 1)
            out = model.forward(x_)
            #MSE with target shift
            loss:Tensor = (out[:,:L,:] - y).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses = 0.98*losses + 0.02*loss.item()
            bar.set_postfix_str(f'loss: {losses: .4f}, lam: {model.lam.item(): 7.1e}')
            
    print(f'step 1: {losses: .6f}')
    
    
    train_y = train_x + torch.randn([num_sample//reduce, L, dim], device=device).tile(reduce,1,1)*0.1 
    train_loader=DataLoader(torch.cat([train_x,train_y],dim=1), batch_size=Bs, shuffle=True)
    
    bar = tqdm(range(num_epoch[1]), ncols=100)
    for epoch in bar:
        bar.set_description_str(f'epoch: {epoch}/{num_epoch[1]}')
        lr_schedu.step()
        for i, x in enumerate(train_loader):
            lr_schedu.step()
            y = x[:,1:]
            out = model.forward(x,)
            #MSE with target shift
            loss:Tensor = (out[:,:L,:] - y).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses = 0.98*losses + 0.02*loss.item()
            bar.set_postfix_str(f'loss: {losses: .4f}, lam: {model.lam.item(): 7.1e}')
    print(f'step 2: {losses: .6f}')
    
        
