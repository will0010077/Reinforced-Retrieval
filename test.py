from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline
import sys
from transformers import AutoTokenizer
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import top_k_sparse

lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to('cpu')
lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'])
k=32

# example='Breaking Bad is an American crime drama television series created and produced by Vince Gilligan for AMC. Set and filmed in Albuquerque, New Mexico, the series follows Walter White (Bryan Cranston), an underpaid, dispirited high-school chemistry teacher struggling with a recent diagnosis of stage-three lung cancer. White turns to a life of crime and partners with a former student, Jesse Pinkman (Aaron Paul), to produce and distribute methamphetamine to secure his family\'s financial future before he dies, while navigating the dangers of the criminal underworld. Breaking Bad premiered on AMC on January 20, 2008, and concluded on September 29, 2013, after five seasons consisting of 62 episodes.'
example='''Advanced RAG brings targeted enhancements to address the shortcomings of Naive RAG. It aims to improve retrieval quality by implementing both pre-retrieval and post-retrieval strategies. To resolve indexing challenges, Advanced RAG uses refined indexing techniques, such as a sliding window approach, fine-grained text segmentation, and the inclusion of metadata. Furthermore, it integrates various optimization methods to make the retrieval process more efficient.

Pre-retrieval Process: At this stage, the main emphasis is on refining the indexing framework and the initial query. The purpose of optimizing indexing is to improve the quality of the indexed content. This process includes several strategies, such as increasing data granularity, refining index architectures, incorporating metadata, optimizing alignment, and employing mixed retrieval techniques. On the other hand, the objective of query optimization is to clarify and adjust the user's original question to make it more suitable for the retrieval task. Common techniques for this include query rewriting, query transformation, query expansion, and other related methods.

Post-Retrieval Process: After retrieving the relevant context, it is essential to integrate it with the query effectively. The main strategies in this stage include re-ranking chunks and context compression. Re-ranking involves reorganizing the retrieved information to place the most relevant content at the forefront of the prompt. This approach is utilized in frameworks like LlamaIndex2, LangChain3, and HayStack. Directly feeding all retrieved documents into LLMs can cause information overload, where key details are obscured by irrelevant content. To address this, post-retrieval efforts focus on selecting the most crucial information, highlighting essential sections, and compressing the context to ensure the key details are not lost.'''


while True:
    example = input()
    tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
    z = lex_MAE_retriver.forward(tokens)
    z = top_k_sparse(z, k)[0]
    for i, v in sorted(zip(lex_MAE_retriver.tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
        print(f'{i}:{v} ',end='')
    print('=============================================================')
# example='''The man played with the child while the dog chased the cat'''
# tokens = lex_MAE_retriver.tokenizer(example, return_tensors='pt')
# z = lex_MAE_retriver.forward(tokens, output_soft=False)
# z = top_k_sparse(z, k)[0]
# count=0
# for i, v in sorted(zip(lex_MAE_retriver.tokenizer.convert_ids_to_tokens(z.coalesce().indices()[0]), z.coalesce().values()), key=lambda x:x[1], reverse=True):
#     print(f'{i:8s}: {v:.2f}, ',end='')
#     count+=1
#     if count%8==0:
#         print()

# exit(0)


# torch.save(a,'tensor.pt')
# torch.save(b,'sparse_tensor.pt')