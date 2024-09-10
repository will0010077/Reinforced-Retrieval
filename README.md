# Prefix-Tuning-with-RL-Retriever

## enviroment prepare
install requirements

```pip install -r requirements.txt```

pre-download model weight

```python save_pretrained.py```

### config
All the config is in config.py

## data preprocess
you should run "dataclean" first

```python data_preprocess.py```

## train retriever
```python train_ret_1.py```

```python train_ret_2.py```

## data preprocess
then run "process Q-A-Doc"

```python data_preprocess.py```

## pre-train encoder
```python train_enc.py```

## train agent
```python train_RL_multi.py```

## experimental result
```python Env_inference.py```
