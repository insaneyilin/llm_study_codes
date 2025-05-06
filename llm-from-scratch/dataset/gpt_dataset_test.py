import os
import sys
import unittest

import tiktoken
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dataset.gpt_dataset import (DEFAULT_GPT_DATALOADER_CONFIG, GPTDatasetV1,
                                 GPTDatasetV2, create_gpt_dataloader_v1,
                                 create_gpt_dataloader_v2)


class TestGPTDatasetV1(unittest.TestCase):
    """Test cases for GPTDatasetV1 class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_text = "Hello world! This is a test text for GPT dataset."
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_length = 10
        self.stride = 5
        self.dataset = GPTDatasetV1(self.test_text, self.tokenizer,
                                    self.max_length, self.stride)

    def test_init(self):
        """Test dataset initialization."""
        self.assertIsInstance(self.dataset.tokenizer, tiktoken.Encoding)
        self.assertIsInstance(self.dataset.input_ids, list)
        self.assertIsInstance(self.dataset.target_ids, list)
        self.assertEqual(len(self.dataset.input_ids),
                         len(self.dataset.target_ids))

        # Check that we have the expected number of chunks
        token_ids = self.tokenizer.encode(self.test_text,
                                          allowed_special={"<|endoftext|>"})
        expected_chunks = max(
            0, (len(token_ids) - self.max_length) // self.stride + 1)
        self.assertEqual(len(self.dataset.input_ids), expected_chunks)

    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.dataset), len(self.dataset.input_ids))

    def test_getitem(self):
        """Test __getitem__ method."""
        if len(self.dataset) > 0:
            item = self.dataset[0]
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            input_ids, target_ids = item
            self.assertIsInstance(input_ids, torch.Tensor)
            self.assertIsInstance(target_ids, torch.Tensor)
            self.assertEqual(len(input_ids), self.max_length)
            self.assertEqual(len(target_ids), self.max_length)

            # Check that targets are inputs shifted by one
            self.assertTrue(torch.equal(input_ids[1:], target_ids[:-1]))

    def test_empty_text(self):
        """Test dataset with empty text."""
        empty_dataset = GPTDatasetV1("", self.tokenizer, self.max_length,
                                     self.stride)
        self.assertEqual(len(empty_dataset), 0)


class TestCreateGPTDataloaderV1(unittest.TestCase):
    """Test cases for create_gpt_dataloader_v1 function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # * 20 in case of the text is too short(may cause dataloader error) for the default config.
        self.test_text = "Hello world! This is a test text for GPT dataloader." * 20

    def test_create_dataloader_default_config(self):
        """Test creating dataloader with default config."""
        dataloader = create_gpt_dataloader_v1(self.test_text)
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

        # Check default config values
        self.assertEqual(dataloader.batch_size,
                         DEFAULT_GPT_DATALOADER_CONFIG['batch_size'])
        self.assertEqual(dataloader.num_workers,
                         DEFAULT_GPT_DATALOADER_CONFIG['num_workers'])

    def test_create_dataloader_custom_config(self):
        """Test creating dataloader with custom config."""
        custom_config = {
            'batch_size': 8,
            'max_length': 128,
            'stride': 64,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 2
        }
        dataloader = create_gpt_dataloader_v1(self.test_text, custom_config)
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

        # Check custom config values
        self.assertEqual(dataloader.batch_size, custom_config['batch_size'])
        self.assertEqual(dataloader.num_workers, custom_config['num_workers'])

    def test_dataloader_iteration(self):
        """Test iterating through the dataloader."""
        config = {
            'batch_size': 2,
            'max_length': 10,
            'stride': 5,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 0
        }
        dataloader = create_gpt_dataloader_v1(self.test_text, config)

        # Try to get one batch
        for batch in dataloader:
            inputs, targets = batch
            self.assertIsInstance(inputs, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)
            self.assertEqual(inputs.shape[0],
                             min(2, len(dataloader.dataset)))  # Batch size
            self.assertEqual(inputs.shape[1],
                             config['max_length'])  # Sequence length
            break


class TestCreateGPTDataloaderV2(unittest.TestCase):
    """Test cases for create_gpt_dataloader_v2 function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.txt_file_paths = [
            "test_data/input1.txt",
            "test_data/input2.txt",
        ]

    def test_dataloader_iteration(self):
        """Test iterating through the dataloader."""
        config = {
            'batch_size': 2,
            'max_length': 10,
            'stride': 5,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 0
        }
        dataloader = create_gpt_dataloader_v2(self.txt_file_paths, config)

        # Try to get one batch
        for batch in dataloader:
            inputs, targets = batch
            print(inputs.shape)
            print(targets.shape)
            print(inputs)
            print(targets)
            self.assertIsInstance(inputs, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)
            self.assertEqual(inputs.shape[0],
                             min(2, len(dataloader.dataset)))  # Batch size
            self.assertEqual(inputs.shape[1],
                             config['max_length'])  # Sequence length
            break


if __name__ == '__main__':
    unittest.main()
