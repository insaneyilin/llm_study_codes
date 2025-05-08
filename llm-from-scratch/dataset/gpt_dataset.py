"""
GPT dataset module for training language models.

This module provides dataset and dataloader implementations for training
GPT-like models using a sliding window approach for next token prediction.
"""

import json
import os

import numpy as np
import tiktoken
import torch
from tqdm import tqdm

DEFAULT_GPT_DATALOADER_CONFIG = {
    'batch_size': 4,
    'max_length': 256,
    'stride': 128,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 0
}


class GPTDatasetV1(torch.utils.data.Dataset):
    """Dataset for training GPT-like models with a sliding window approach.

    This dataset tokenizes text and creates input-target pairs where the target
    is the input shifted by one token (next token prediction).

    Attributes:
        tokenizer: The tokenizer used to encode the text.
        input_ids: List of input token sequences.
        target_ids: List of target token sequences (shifted by one).
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """Initialize the dataset with text and tokenization parameters.

        Args:
            txt (str): The input text to tokenize and process.
            tokenizer: The tokenizer to use for encoding the text.
            max_length (int): Maximum sequence length for each chunk.
            stride (int): Number of tokens to slide the window by for each chunk.
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A pair of (input_ids, target_ids) tensors.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_gpt_dataloader_v1(txt, config=DEFAULT_GPT_DATALOADER_CONFIG):
    """Create a DataLoader for GPT training.

    This function handles the entire process of creating a ready-to-use DataLoader
    for training GPT models, including tokenization and dataset creation. We use
    the default tokenizer for GPT-2.

    Args:
        txt (str): The input text to use for training.
        config (dict, optional): Configuration dictionary with the following keys:
            - batch_size (int): Number of samples per batch. Defaults to 4.
            - max_length (int): Maximum sequence length. Defaults to 256.
            - stride (int): Stride for the sliding window. Defaults to 128.
            - shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            - drop_last (bool): Whether to drop the last incomplete batch. Defaults to True.
            - num_workers (int): Number of worker processes for data loading. Defaults to 0.

    Returns:
        DataLoader: A PyTorch DataLoader ready for training.
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, config['max_length'],
                           config['stride'])

    # Create dataloader
    persistent_workers = config['num_workers'] > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        drop_last=config['drop_last'],
        num_workers=config['num_workers'],
        persistent_workers=persistent_workers)

    return dataloader


class GPTDatasetV2(torch.utils.data.Dataset):
    """Dataset for training GPT-like models with multiple text files.

    This dataset processes multiple text files efficiently by pre-computing token indices
    and loading file segments only when needed.

    Attributes:
        tokenizer: The tokenizer used to encode the text.
        file_paths: List of paths to text files.
        file_token_indices: List of token indices for each file.
        max_length: Maximum sequence length for each chunk.
        stride: Number of tokens to slide the window by for each chunk.
        total_samples: Total number of samples across all files.
        file_cache: Cache for storing recently accessed file token data.
    """

    def __init__(self,
                 file_paths,
                 tokenizer,
                 max_length,
                 stride,
                 cache_size=3):
        """Initialize the dataset with file paths and tokenization parameters.

        Args:
            file_paths (list): List of paths to text files.
            tokenizer: The tokenizer to use for encoding the text.
            max_length (int): Maximum sequence length for each chunk.
            stride (int): Number of tokens to slide the window by for each chunk.
            cache_size (int): Number of tokenized files to keep in memory cache.
        """
        self.tokenizer = tokenizer
        self.file_paths = file_paths
        self.max_length = max_length
        self.stride = stride

        # Pre-compute token indices for each file
        self.file_token_indices = []
        self.total_samples = 0

        print(f"Indexing {len(file_paths)} files...")
        for i, file_path in enumerate(file_paths):
            print(f"Indexing file {i+1}/{len(file_paths)}: {file_path}")
            # Get file size without loading content
            file_size = os.path.getsize(file_path)

            # For very large files, estimate token count based on file size
            # Average bytes per token for English text with GPT-2 tokenizer is ~4
            estimated_tokens = file_size // 4

            # Calculate estimated number of samples
            estimated_samples = max(
                0, (estimated_tokens - max_length) // stride + 1)

            # Store file info
            self.file_token_indices.append({
                'path': file_path,
                'start_idx': self.total_samples,
                'estimated_samples': estimated_samples,
                'estimated_tokens': estimated_tokens
            })

            self.total_samples += estimated_samples

        # File content cache - store tokenized content of recently used files
        self.file_cache = {}
        self.cache_size = cache_size

        # Add a token segment cache to avoid re-tokenizing the same file segments
        self.segment_cache = {}
        self.max_segment_cache = cache_size * 5  # Allow more segment caches than file caches

    def __len__(self):
        """Return the total number of samples across all files."""
        return self.total_samples

    def _get_file_segment(self, file_idx, start_pos, length):
        """Get a segment of tokens from a file, using cache if available."""
        cache_key = (file_idx, start_pos, length)

        # Check if segment is in cache
        if cache_key in self.segment_cache:
            return self.segment_cache[cache_key]

        # If we need to load the file
        if file_idx not in self.file_cache:
            file_path = self.file_paths[file_idx]
            file_info = self.file_token_indices[file_idx]
            estimated_tokens = file_info['estimated_tokens']

            # For very large files, only read and tokenize the needed segment
            if estimated_tokens > 1_000_000:  # Threshold for "large file"
                # Calculate byte offsets (approximate)
                bytes_per_token = 4  # Estimate
                approx_start_byte = max(0, start_pos * bytes_per_token -
                                        1000)  # Add buffer
                approx_length_bytes = (length +
                                       100) * bytes_per_token  # Add buffer

                with open(file_path, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    # Seek to approximate position
                    f.seek(approx_start_byte)
                    # Skip to the next complete line
                    if approx_start_byte > 0:
                        f.readline()
                    # Read enough text to cover our needs
                    text_segment = f.read(approx_length_bytes)

                # Tokenize just this segment
                tokens = self.tokenizer.encode(
                    text_segment, allowed_special={"<|endoftext|>"})

                # Adjust for potential offset issues
                if len(tokens) < length + 1:
                    # If we didn't get enough tokens, read more
                    with open(file_path,
                              'r',
                              encoding='utf-8',
                              errors='ignore') as f:
                        text = f.read()
                    tokens = self.tokenizer.encode(
                        text, allowed_special={"<|endoftext|>"})

                # Store in segment cache
                segment = tokens[start_pos:start_pos + length + 1]
                self.segment_cache[cache_key] = segment

                # Manage segment cache size
                if len(self.segment_cache) > self.max_segment_cache:
                    self.segment_cache.pop(next(iter(self.segment_cache)))

                return segment
            else:
                # For smaller files, load the entire file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        txt = f.read()
                except UnicodeDecodeError:
                    # Try with a different encoding if utf-8 fails
                    with open(file_path, 'r', encoding='latin-1') as f:
                        txt = f.read()

                # Tokenize
                tokens = self.tokenizer.encode(
                    txt, allowed_special={"<|endoftext|>"})

                # Update file cache
                if len(self.file_cache) >= self.cache_size:
                    # Remove least recently used file
                    self.file_cache.pop(next(iter(self.file_cache)))
                self.file_cache[file_idx] = tokens
        else:
            # File is already in cache
            tokens = self.file_cache[file_idx]

        # Get the segment
        segment = tokens[start_pos:start_pos + length + 1]

        # Store in segment cache
        self.segment_cache[cache_key] = segment

        # Manage segment cache size
        if len(self.segment_cache) > self.max_segment_cache:
            self.segment_cache.pop(next(iter(self.segment_cache)))

        return segment

    def _find_file_index(self, idx):
        """Find which file contains the sample at the given index using binary search."""
        left, right = 0, len(self.file_token_indices) - 1

        while left <= right:
            mid = (left + right) // 2
            if mid == len(
                    self.file_token_indices
            ) - 1 or idx < self.file_token_indices[mid + 1]['start_idx']:
                if idx >= self.file_token_indices[mid]['start_idx']:
                    return mid, idx - self.file_token_indices[mid]['start_idx']
                right = mid - 1
            else:
                left = mid + 1

        # Fallback (should not reach here with valid indices)
        for i, file_info in enumerate(self.file_token_indices):
            if i == len(self.file_token_indices
                        ) - 1 or idx < self.file_token_indices[i +
                                                               1]['start_idx']:
                return i, idx - file_info['start_idx']

        return len(self.file_token_indices
                   ) - 1, idx - self.file_token_indices[-1]['start_idx']

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A pair of (input_ids, target_ids) tensors.
        """
        # Find which file contains this index
        file_idx, local_idx = self._find_file_index(idx)

        # Calculate start position based on local index
        start_pos = local_idx * self.stride

        # Get the segment directly (including input and target)
        segment = self._get_file_segment(file_idx, start_pos,
                                         self.max_length + 1)

        # Handle edge case - if segment is too short
        if len(segment) < self.max_length + 1:
            # Pad if necessary
            if len(segment) <= 1:  # Empty or single token
                input_chunk = [0] * self.max_length
                target_chunk = [0] * self.max_length
            else:
                input_chunk = segment[:-1]
                target_chunk = segment[1:]

                # Pad if still too short
                if len(input_chunk) < self.max_length:
                    input_chunk = input_chunk + [0] * (self.max_length -
                                                       len(input_chunk))
                if len(target_chunk) < self.max_length:
                    target_chunk = target_chunk + [0] * (self.max_length -
                                                         len(target_chunk))
        else:
            # Normal case - we have enough tokens
            input_chunk = segment[:self.max_length]
            target_chunk = segment[1:self.max_length + 1]

        # Convert to tensors
        input_tensor = torch.tensor(input_chunk)
        target_tensor = torch.tensor(target_chunk)

        return input_tensor, target_tensor


def create_gpt_dataloader_v2(file_paths,
                             config=DEFAULT_GPT_DATALOADER_CONFIG,
                             cache_size=3):
    """Create a DataLoader for GPT training from multiple text files.

    This function handles the entire process of creating a ready-to-use DataLoader
    for training GPT models from multiple text files, including tokenization and
    dataset creation.

    Args:
        file_paths (list): List of paths to text files to use for training.
        config (dict, optional): Configuration dictionary with the following keys:
            - batch_size (int): Number of samples per batch. Defaults to 4.
            - max_length (int): Maximum sequence length. Defaults to 256.
            - stride (int): Stride for the sliding window. Defaults to 128.
            - shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            - drop_last (bool): Whether to drop the last incomplete batch. Defaults to True.
            - num_workers (int): Number of worker processes for data loading. Defaults to 0.
        cache_size (int): Number of tokenized files to keep in memory cache. Defaults to 3.

    Returns:
        DataLoader: A PyTorch DataLoader ready for training.
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV2(file_paths,
                           tokenizer,
                           config['max_length'],
                           config['stride'],
                           cache_size=cache_size)

    # Create dataloader
    persistent_workers = config['num_workers'] > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        drop_last=config['drop_last'],
        num_workers=config['num_workers'],
        persistent_workers=persistent_workers)

    return dataloader


# Like GPTDatasetV1, but for multiple text files.
def preprocess_txt_files_to_bin(train_file_paths, val_file_paths, tokenizer,
                                max_length, stride, output_dir):
    """
    Preprocess multiple text files into binary format for efficient training.

    Args:
        train_file_paths: List of paths to training text files
        val_file_paths: List of paths to validation text files
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        stride: Stride for sliding window
        output_dir: Directory to save binary files
    """
    dtype = np.uint16  # GPT2 vocab_size 50257 is < 2**16

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def load_txt(file_path):
        """Helper function to load text from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def process_files(file_paths, output_filename):
        """Helper function to process a list of files into a binary file."""
        # First pass: calculate total number of sequences
        total_sequences = 0
        for file_path in tqdm(file_paths, desc="Counting sequences"):
            txt = load_txt(file_path)
            token_ids = tokenizer.encode(txt,
                                         allowed_special={"<|endoftext|>"})
            # Calculate how many sequences this file will produce
            if len(token_ids) >= max_length:
                total_sequences += (len(token_ids) - max_length) // stride + 1

        if total_sequences == 0:
            raise ValueError(
                "No sequences generated - check your input files and parameters"
            )

        # Initialize memory-mapped array
        output_shape = (total_sequences, max_length)
        arr = np.memmap(os.path.join(output_dir, output_filename),
                        dtype=dtype,
                        mode='w+',
                        shape=output_shape)

        # Second pass: actually process and save the data
        seq_idx = 0
        for file_path in tqdm(file_paths, desc="Processing files"):
            txt = load_txt(file_path)
            token_ids = tokenizer.encode(txt,
                                         allowed_special={"<|endoftext|>"})

            # Generate sequences with sliding window
            for i in range(0, len(token_ids) - max_length + 1, stride):
                chunk = token_ids[i:i + max_length]
                arr[seq_idx] = chunk
                seq_idx += 1

        # Flush changes to disk
        arr.flush()
        del arr
        return total_sequences

    # Process training and validation files
    print("Processing training files...")
    train_seq_num = process_files(train_file_paths, "train.bin")

    print("Processing validation files...")
    val_seq_num = process_files(val_file_paths, "val.bin")

    bin_data_meta_json_filepath = os.path.join(output_dir,
                                               "bin_data_meta.json")
    meta_info = {
        "description": "Preprocessed binary data for GPT training",
        "train_file_paths": train_file_paths,
        "val_file_paths": val_file_paths,
        "output_dir": output_dir,
        "max_length": max_length,
        "stride": stride,
        "dtype": str(dtype),
        "train_file": "train.bin",
        "val_file": "val.bin",
        "train_seq_num": train_seq_num,
        "val_seq_num": val_seq_num,
        "tokenizer_name": tokenizer.name,
    }
    with open(bin_data_meta_json_filepath, 'w') as f:
        json.dump(meta_info, f, indent=4)


class GPTDatasetBinary(torch.utils.data.Dataset):
    """Dataset that reads from preprocessed binary files."""

    def __init__(self, bin_file_path, max_length):
        self.data = np.memmap(bin_file_path, dtype=np.uint16, mode='r')
        self.max_length = max_length
        # Reshape to (num_sequences, max_length)
        self.num_sequences = len(self.data) // max_length
        self.data = self.data.reshape((self.num_sequences, max_length))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Input is all tokens except last, target is all tokens except first
        x = torch.from_numpy(self.data[idx][:-1].astype(np.int64))
        y = torch.from_numpy(self.data[idx][1:].astype(np.int64))
        return x, y


def create_gpt_bin_dataloader(bin_file_path,
                              config=DEFAULT_GPT_DATALOADER_CONFIG):
    # Create dataset
    dataset = GPTDatasetBinary(bin_file_path, config['max_length'])

    # Create dataloader
    persistent_workers = config['num_workers'] > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        drop_last=config['drop_last'],
        num_workers=config['num_workers'],
        persistent_workers=persistent_workers)

    return dataloader


class GPTDatasetBinaryV2(torch.utils.data.Dataset):
    """Dataset that reads from preprocessed binary files."""

    def __init__(self, bin_file_path, max_length):
        self.bin_file_path = bin_file_path
        self.max_length = max_length

        arr = np.memmap(bin_file_path, dtype=np.uint16, mode='r')
        self.num_sequences = len(arr) // max_length
        del arr

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        arr = np.memmap(self.bin_file_path, dtype=np.uint16, mode='r')
        arr = arr.reshape((self.num_sequences, self.max_length))

        sequence = arr[idx]

        x = torch.from_numpy(sequence[:-1].astype(np.int64))
        y = torch.from_numpy(sequence[1:].astype(np.int64))
        return x, y


def create_gpt_bin_dataloader_v2(bin_file_path,
                                 config=DEFAULT_GPT_DATALOADER_CONFIG):
    # Create dataset
    dataset = GPTDatasetBinaryV2(bin_file_path, config['max_length'])

    # Create dataloader
    persistent_workers = config['num_workers'] > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        drop_last=config['drop_last'],
        num_workers=config['num_workers'],
        persistent_workers=persistent_workers)

    return dataloader
