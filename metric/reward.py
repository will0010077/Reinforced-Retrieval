import nltk
from nltk.translate.bleu_score import sentence_bleu
from typing import List
import torch
from bert_score import score
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np



nltk.download('punkt')


def Bert_score(a: List[str], b: List[str]) -> List[float]:
    """
    Calculate the BERT score for each pair of sentences in a and b.
    
    :param a: List of generated answers.
    :param b: List of reference answers.
    :return: List of BERT scores for each pair of answers.
    """
    P, R, F1 = score(a, b, lang="en", model_type='bert-base-uncased', device = 1)
    return F1.tolist()

def BLEU_score(a: List[str], b: List[str]) -> List[float]:
    """
    Calculate the BLEU score for each pair of sentences in a and b.
    
    :param a: List of generated answers.
    :param b: List of reference answers.
    :return: List of BLEU scores for each pair of answers.
    """
    bleu_scores = []
    for gen, ref in zip(a, b):
        ref_tokens = nltk.word_tokenize(ref)
        gen_tokens = nltk.word_tokenize(gen)
        bleu = sentence_bleu([ref_tokens], gen_tokens)
        bleu_scores.append(bleu)
    return bleu_scores




class HalluScore:
    def __init__(self, model_name: str = 'gpt2'):
       
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def get_embeddings(self,texts):
        
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


    def score(self,generated_texts, reference_texts):
        
        
        generated_embeddings = self.get_embeddings(generated_texts)
        reference_embeddings = self.get_embeddings(reference_texts)
        cosine_similarities = torch.nn.functional.cosine_similarity(generated_embeddings, reference_embeddings)
        hallucination_score = torch.ones_like(cosine_similarities) - cosine_similarities
        
        return hallucination_score.tolist()



if __name__=='__main__':
    a=['this is a dog','i love dongdong','i love ml!']
    b=['this is a dog','i love study','i love dl!']


    reward_bleu=BLEU_score(a,b)
    
    reward_bert=Bert_score(a,b)
    
    hallu_score=HalluScore()
    reward_hallu_score = hallu_score.score(a, b)
    print(f'output:{a}\ntarget:{b}')
    print(f'bleu:{reward_bleu} \nbert:{reward_bert} \nhallu: {reward_hallu_score}')