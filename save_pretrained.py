from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, RobertaModel, RobertaForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.save_pretrained("huggingface/roberta_head/")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.save_pretrained("huggingface/roberta_head/")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.save_pretrained("huggingface/bert/")
# model = AutoModel.from_pretrained("bert-base-uncased")
# model.save_pretrained("huggingface/bert/")

# token = "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"
# model_dir = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token=token)
# tokenizer.save_pretrained("huggingface/llama2/")
# model=AutoModelForCausalLM.from_pretrained(model_dir, token=token, device_map='cpu', use_cache=True, torch_dtype = torch.bfloat16)
# model.save_pretrained("huggingface/llama2/")