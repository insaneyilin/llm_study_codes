import os
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dataset.gpt_dataset import (GPTDatasetBinary, create_gpt_bin_dataloader,
                                 preprocess_txt_files_to_bin)

DATA_DIR = 'data/'
train_file_paths = ['combined_19_trimmed.txt']
val_file_paths = ['the-verdict.txt']

train_file_paths = [os.path.join(DATA_DIR, file) for file in train_file_paths]
val_file_paths = [os.path.join(DATA_DIR, file) for file in val_file_paths]

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

# use same stride as max_length
tokenizer = tiktoken.get_encoding("gpt2")

preprocess_txt_files_to_bin(train_file_paths,
                            val_file_paths,
                            tokenizer,
                            max_length=GPT_CONFIG_124M['context_length'],
                            stride=GPT_CONFIG_124M['context_length'],
                            output_dir='data/gpt_bin_dataset')

# gpt_dataset = GPTDatasetBinary('data/gpt_bin_dataset/val.bin', GPT_CONFIG_124M['context_length'])

# for i in range(5):
#     input_token_ids = gpt_dataset[i][0]
#     target_token_ids = gpt_dataset[i][1]
#     print(f"Input {i}:")
#     print(tokenizer.decode(input_token_ids.tolist()))
#     print(f"Target {i}:")
#     print(tokenizer.decode(target_token_ids.tolist()))

train_loader_cfg = {
    "batch_size": 8,
    "max_length": GPT_CONFIG_124M["context_length"],
    "stride": GPT_CONFIG_124M["context_length"],
    "drop_last": True,
    "shuffle": True,
    "num_workers": 0
}

data_loader = create_gpt_bin_dataloader('data/gpt_bin_dataset/train.bin',
                                        train_loader_cfg)
for i, (input_token_ids, target_token_ids) in enumerate(data_loader):
    if i == 0:
        print(f"Input {i}:")
        print(tokenizer.decode(input_token_ids[0].tolist()))
        print(f"Target {i}:")
        print(tokenizer.decode(target_token_ids[0].tolist()))
    if i == 1:
        break
