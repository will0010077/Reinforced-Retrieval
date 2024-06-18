from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2", torch_dtype = torch.bfloat16,device='cuda:1')
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
stop_id = [tokenizer.eos_token_id]
for k,v in tokenizer.vocab.items():
    if "\"" in k:
        stop_id.append(v)
print(stop_id)
message = "for the question \"who is the girl locked up in don't breathe\". The answer is: \"Rocky and Alex evade the Blind Man and hurry to the basement. There, they are shocked to find a restrained, gagged woman in a homemade padded cell.\" the answer is correct (True/False): "
LMout = pipe(message, num_return_sequences=10, top_p = 0.5, max_new_tokens = 8, no_repeat_ngram_size = 4, eos_token_id = stop_id)
for i in LMout:
    print(i['generated_text'][len(message)-1:])

exit()
if __name__=='__main__':
#     data=torch.load('data/vecs_reduced_5000000.pt') ## shape:(N,d)
#     print('converting...')
#     data = unbind_sparse(data)
#     for i, d in enumerate(data):
#         if (i==d.indices()).float().mean() != 1.:
#             print(d)
#     exit()
#     a = torch.randn([32,10]).to_sparse()
#     print(a.sparse_dim(), a.dense_dim())
#     a.sparse_resize_([64,10], a.sparse_dim(), a.dense_dim())
#     print(a)
#     exit()
    
#     s=time()    
    a = torch.load('data/vecs_reduced_10000000.pt',mmap=True)
    a= top_k_sparse(a, 64)
    torch.save('data/vecs_reduced_10000000.pt')
    exit()
#     print(time()-s)
    
#     print(a[10000])
    
#     s=time()
#     # a = restore_batched_list(a)
#     a = unbind_sparse(a)
    
#     print(time()-s)
#     print(a[10000])
#     exit()
#     s=time()
#     # a = split_list_to_batch(a, bs=32)
#     print(time()-s)
#     s=time()
#     torch.save(a, 'data/vecs_reduced_5000000.pt')
#     print(time()-s)
#     s=time()
#     exit()
import sys
from transformers import AutoTokenizer
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import top_k_sparse

lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to('cpu')
lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever878.pt', map_location='cpu')['enc_model_state_dict'])
k=256

# example='Breaking Bad is an American crime drama television series created and produced by Vince Gilligan for AMC. Set and filmed in Albuquerque, New Mexico, the series follows Walter White (Bryan Cranston), an underpaid, dispirited high-school chemistry teacher struggling with a recent diagnosis of stage-three lung cancer. White turns to a life of crime and partners with a former student, Jesse Pinkman (Aaron Paul), to produce and distribute methamphetamine to secure his family\'s financial future before he dies, while navigating the dangers of the criminal underworld. Breaking Bad premiered on AMC on January 20, 2008, and concluded on September 29, 2013, after five seasons consisting of 62 episodes.'
example='''Obama's first-term actions addressed the global financial crisis and included a major stimulus package to guide the economy in recovering from the Great Recession, a partial extension of George W. Bush's tax cuts, legislation to reform health care, a major financial regulation reform bill, and the end of a major U.S. military presence in Iraq. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He ordered the counterterrorism raid which killed Osama bin Laden and downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, while encouraging greater reliance on host-government militaries. Obama also ordered military involvement in Libya in order to implement UN Security Council Resolution 1973, contributing to the overthrow of Muammar Gaddafi.'''


tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
z = lex_MAE_retriver.forward(tokens)
z = top_k_sparse(z, k)[0]
for i, v in sorted(zip(lex_MAE_retriver.tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
    print(f'{i}:{v:.3f}, ',end='')
print('=============================================================')
example='''The man played with the child while the dog chased the cat'''
tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
z = lex_MAE_retriver.forward(tokens, output_soft=False)
z = top_k_sparse(z, k)[0]
count=0
for i, v in sorted(zip(lex_MAE_retriver.tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
    print(f'{i:8s}: {v:.2f}, ',end='')
    count+=1
    if count%8==0:
        print()

exit(0)


# torch.save(a,'tensor.pt')
# torch.save(b,'sparse_tensor.pt')