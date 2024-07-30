import nltk
from nltk.translate.bleu_score import sentence_bleu
from typing import List
import torch
from bert_score import score
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from bert_score.utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer,
                    lang2model, model2layers, sent_encode)
from collections import defaultdict
import evaluate

class metric(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        nltk.download('punkt')

        model_type='bert-base-uncased'
        self.tokenizer = get_tokenizer(model_type, True)
        num_layers = model2layers[model_type]
        self.model = get_model(model_type, num_layers, False)
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load("bleu")

    def Bert_score(self, cands: List[str], refs: List[str]) -> tuple[torch.Tensor]:
        """
        Calculate the BERT score for each pair of sentences in cands and refs.
        
        :param cands: List of generated answers.
        :param refs: List of reference answers.
        :return: List of BERT scores for each pair of answers.
        """
        # P, R, F1 = score(a, b, lang="en", model_type='bert-base-uncased', device = None)
        
        
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[self.tokenizer.sep_token_id] = 0
        idf_dict[self.tokenizer.cls_token_id] = 0
        
        all_preds = bert_cos_score_idf(
            self.model,
            refs,
            cands,
            self.tokenizer,
            idf_dict,
            device=self.model.device,
        ).cpu()
        P, R, F1  = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
        score = F1
        if any(score==0.):
            raise ValueError
        return score.unbind()

    def BLEU_1_score(self, cands: List[str], refs: List[str]) -> List[float]:
        """
        Calculate the BLEU score for each pair of sentences in a and b.
        
        :param a: List of generated answers.
        :param b: List of reference answers.
        :return: List of BLEU scores for each pair of answers.
        """
        bleu_scores = []
        for gen, ref in zip(cands, refs):
            ref_tokens = nltk.word_tokenize(ref)
            gen_tokens = nltk.word_tokenize(gen)
            bleu = sentence_bleu([ref_tokens], gen_tokens, weights=[1.])
            bleu_scores.append(bleu)
        return bleu_scores

    def ROUGE_score(self, pred, ref):
        ''' return R-1, R-2, R-L'''
        results = self.rouge.compute(predictions=pred, references=ref, use_aggregator=False, rouge_types=["rouge1","rouge2", 'rougeL'])
        return results["rouge1"], results["rouge2"], results["rougeL"]
        #{'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}


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


