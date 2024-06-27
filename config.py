class Config(dict):
    def __init__(self, **entries):
        super().__init__(entries)
        self.__dict__.update(entries)
        
train_config = Config(
    max_epoch=6,
    spilt=0.95,
    num_samples=None,
    lm_lr=1.e-5,
    agent_lr=0.e-5,
    betas=[0.8, 0.998],
    weight_decay=0.01,
    topk=1,
    load_E=True,
    use_prefix=True
)

enc_config = Config(
    enc_lr=3.e-4,
    groups=1,
    num_layers=16,
    num_prefix=8
)

enc_size_config = Config(
    num_hidden_layers=6,
    num_attention_heads=8,
    hidden_size=768,
    intermediate_size=3072
)

agent_size_config = Config(
    num_hidden_layers=4,
    num_attention_heads=12,
    hidden_size=768,
    intermediate_size=3072,
    max_position_embeddings = 1024,
)

lex = Config(
    pre_lr=5.e-5,
    fine_lr=1.e-5,
    betas=[0.9, 0.999],
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
    num_doc=200000,
    num_samples=100000,
    windows=192,
    step=128
)

generate_config = Config(
    no_repeat_ngram_size=6,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    num_beams=1,
    bad_words_ids=None
)

seed = 87

bert_dir = "huggingface/bert"
LM_dir = "huggingface/llama2"

token = "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"