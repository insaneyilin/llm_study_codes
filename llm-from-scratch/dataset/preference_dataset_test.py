import os
import sys
import unittest

import tiktoken
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from preference_dataset import (PreferenceDataset, decode_tokens_from_batch,
                                format_input)


class TestPreferenceDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.test_data = [
            {
                "instruction": "Edit the following sentence for grammar.",
                "input": "He go to the park every day.",
                "output": "He goes to the park every day.",
                "rejected": "He goes to the stupid park every single day.",
                "chosen": "He goes to the park every day."
            },
            {
                "instruction": "Convert 45 kilometers to meters.",
                "input": "",
                "output": "45 kilometers is 45000 meters.",
                "chosen": "45 kilometers is equivalent to 45000 meters.",
                "rejected": "45 kilometers is 45000 meters."
            },
        ]
        self.dataset = PreferenceDataset(self.test_data, self.tokenizer)

    def test_dataset_length(self):
        """Test that the dataset length matches the input data length."""
        self.assertEqual(len(self.dataset), len(self.test_data))

    def test_dataset_getitem(self):
        """Test that __getitem__ returns the expected format."""
        item = self.dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn('prompt', item)
        self.assertIn('chosen', item)
        self.assertIn('rejected', item)
        self.assertIsInstance(item['prompt'], list)
        self.assertIsInstance(item['chosen'], list)
        self.assertIsInstance(item['rejected'], list)

    def test_format_input(self):
        """Test the format_input function."""
        example = self.test_data[0]
        formatted = format_input(example)
        expected = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nEdit the following sentence for grammar.\n\n### Input:\nHe go to the park every day."
        self.assertEqual(formatted, expected)

        # Test with empty input
        example = self.test_data[1]
        formatted = format_input(example)
        expected = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nConvert 45 kilometers to meters."
        self.assertEqual(formatted, expected)

    def test_decode_tokens(self):
        """Test the decode_tokens_from_batch function."""
        # Get a batch by accessing the first item
        item = self.dataset[0]
        chosen_decoded = decode_tokens_from_batch(item['chosen'],
                                                  self.tokenizer)
        rejected_decoded = decode_tokens_from_batch(item['rejected'],
                                                    self.tokenizer)

        # Check that decoded text contains the expected output
        self.assertIn("He goes to the park every day", chosen_decoded)
        self.assertIn("He goes to the stupid park every single day",
                      rejected_decoded)


if __name__ == "__main__":
    unittest.main()
