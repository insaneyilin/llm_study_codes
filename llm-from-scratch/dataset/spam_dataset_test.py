import os
import sys
import unittest

import tiktoken
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dataset.spam_dataset import SpamDataset


class TestSpamDataset(unittest.TestCase):

    def setUp(self):
        self.csv_file = "sms_spam_collection/test.csv"
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_length = 10

    def test_dataset_initialization(self):
        """Test that the dataset can be initialized properly."""
        dataset = SpamDataset(self.csv_file, self.tokenizer, self.max_length)
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)

    def test_getitem(self):
        """Test that __getitem__ returns the expected format."""
        dataset = SpamDataset(self.csv_file, self.tokenizer, self.max_length)
        item = dataset[0]

        # Check that the item is a tuple of (input_ids, label)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)

        # Check input_ids
        input_ids, label = item
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertLessEqual(len(input_ids), self.max_length)

        # Check label
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.shape, torch.Size([]))  # Scalar tensor
        self.assertIn(label.item(), [0, 1])  # Binary classification

    def test_len(self):
        """Test that __len__ returns the correct dataset size."""
        dataset = SpamDataset(self.csv_file, self.tokenizer, self.max_length)
        # We can't know the exact size without reading the file,
        # but we can check that it's positive
        self.assertGreater(len(dataset), 0)

    def test_max_length_truncation(self):
        """Test that sequences are truncated to max_length."""
        dataset = SpamDataset(self.csv_file, self.tokenizer, self.max_length)

        for i in range(min(10, len(dataset))):  # Check first 10 items
            input_ids, _ = dataset[i]
            self.assertLessEqual(len(input_ids), self.max_length)


if __name__ == "__main__":
    unittest.main()
