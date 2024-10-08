from typing import Any
'''All config that you needed in training process'''

class Config(dict):
    def __init__(self, **entries):
        super().__init__(entries)
        self.__dict__.update(entries)
    def __setattr__(self, name: str, value: Any) -> None:
        self.update({name: value})
        return self.__dict__.update({name: value})
    def __setitem__(self, key: Any, value: Any) -> None:
        self.__dict__.update({key: value})
        return super().__setitem__(key, value)
    def copy(self,):
        return Config(self)

train_config = Config(
    max_epoch=6,
    spilt=0.95,
    num_samples=None,
    lm_lr=1.e-5,
    agent_lr=3.e-5,
    agent_head_lr = 5e-4,
    betas=[0.9, 0.96],
    weight_decay=0.01,
    topk=1,
    load_E=True,
    use_prefix=True
)
ppo_config = Config(
    gamma=0.99,
    clip_epsilon=0.2,
    lambd=0.95,
    batch_size=64,
    grad_step=1
)
enc_config = Config(
    enc_lr=5e-6,
    prefix_lr=1.5e-5,
    num_layers=31,
    num_prefix=20
)

enc_size_config = Config(
    num_hidden_layers=12,
    hidden_size=768,
)

agent_size_config = Config(
    max_position_embeddings = 512,
)

lex = Config(
    pre_lr=5.e-5,
    fine_lr=1.e-5,
    betas=[0.9, 0.98],
    weight_decay=0.01,
    share_param=True
)

loader_config = Config(
    batch_size=2,
    shuffle=True,
    num_workers=1,
    persistent_workers=True
)

cluster_config = Config(
    k=3000,
    bs=200000,
    lr=0.5,
    tol=0.05,
    num_search=4,
    k_sparse=64
)

data_config = Config(
    windows=96,
    step=64
)

generate_config = Config(
    no_repeat_ngram_size=8,
    do_sample=True,
    temperature=1,
    top_p=0.9,
    top_k=None,
    num_beams=1,
    bad_words_ids=None
)

seed = 87

bert_dir = "huggingface/bert/"
roberta_dir = "huggingface/roberta/"
LM_dir = "huggingface/llama2/"

token = "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"