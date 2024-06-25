from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenizer.save_pretrained("huggingface/bert/")
token = "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"
model_dir = "meta-llama/Llama-2-7b-chat-hf"
model=AutoModelForCausalLM.from_pretrained(model_dir, token=token, device_map='cpu', use_cache=True, torch_dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token=token)
model.save_pretrained("huggingface/llama2/")
tokenizer.save_pretrained("huggingface/llama2/")