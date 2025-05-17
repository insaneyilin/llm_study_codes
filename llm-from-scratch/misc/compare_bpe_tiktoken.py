import timeit
from importlib.metadata import version

import tiktoken
from bpe_openai_gpt2 import download_vocab, get_encoder
from transformers import GPT2Tokenizer


def print_version_info():
    print("tiktoken version:", version("tiktoken"))
    print("transformers version:", version("transformers"))


def initialize_tokenizers():
    # tiktoken tokenizer
    tik_tokenizer = tiktoken.get_encoding("gpt2")

    # Original OpenAI GPT-2 tokenizer
    download_vocab()
    orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

    # HuggingFace tokenizer
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return tik_tokenizer, orig_tokenizer, hf_tokenizer


def test_encode_decode(tokenizers, text="Hello, world. Is this-- a test?"):
    tik_tokenizer, orig_tokenizer, hf_tokenizer = tokenizers

    print("\nTesting tiktoken:")
    integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("Encoded:", integers)
    print("Decoded:", tik_tokenizer.decode(integers))
    print("Vocabulary size:", tik_tokenizer.n_vocab)

    print("\nTesting original OpenAI tokenizer:")
    integers = orig_tokenizer.encode(text)
    print("Encoded:", integers)
    print("Decoded:", orig_tokenizer.decode(integers))

    print("\nTesting HuggingFace tokenizer:")
    encoded = hf_tokenizer(text)
    print("Encoded:", encoded["input_ids"])
    print("Decoded:", hf_tokenizer.decode(encoded["input_ids"]))


def benchmark_performance(tokenizers, file_path='../data/the-verdict.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    tik_tokenizer, orig_tokenizer, hf_tokenizer = tokenizers

    print("\nBenchmarking original OpenAI tokenizer:")
    print(timeit.timeit(lambda: orig_tokenizer.encode(raw_text), number=10))

    print("\nBenchmarking tiktoken:")
    print(
        timeit.timeit(lambda: tik_tokenizer.encode(
            raw_text, allowed_special={"<|endoftext|>"}),
                      number=10))

    print("\nBenchmarking HuggingFace tokenizer (no truncation):")
    print(timeit.timeit(lambda: hf_tokenizer(raw_text)["input_ids"],
                        number=10))

    print("\nBenchmarking HuggingFace tokenizer (with truncation):")
    print(
        timeit.timeit(lambda: hf_tokenizer(
            raw_text, max_length=5145, truncation=True)["input_ids"],
                      number=10))


def main():
    print_version_info()
    tokenizers = initialize_tokenizers()
    test_encode_decode(tokenizers)
    benchmark_performance(tokenizers)


if __name__ == "__main__":
    main()

# Benchmarking original OpenAI tokenizer:
# 0.05131308315321803

# Benchmarking tiktoken:
# 0.009408916346728802

# Benchmarking HuggingFace tokenizer (no truncation):
# Token indices sequence length is longer than the specified maximum sequence length for this model (5146 > 1024). Running this sequence through the model will result in indexing errors
# 0.12264037504792213

# Benchmarking HuggingFace tokenizer (with truncation):
# 0.11061624996364117
