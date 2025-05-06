import os
import sys
import unittest

import tiktoken
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from instruction_dataset import InstructionDataset, format_input


class TestInstructionDataset(unittest.TestCase):

    def setUp(self):
        # Create test data
        self.test_data = [{
            "instruction": "Test instruction 1",
            "input": "Test input 1",
            "output": "Test output 1"
        }, {
            "instruction": "Test instruction 2",
            "input": "",
            "output": "Test output 2"
        }, {
            "instruction": "Test instruction 3",
            "input": "Test input 3",
            "output": "Test output 3"
        }]
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def test_format_input(self):
        # Test with input
        entry = {"instruction": "Test instruction 1", "input": "Test input 1"}
        formatted = format_input(entry)
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction 1"
            "\n\n### Input:\nTest input 1")
        self.assertEqual(formatted, expected)

        # Test without input
        entry = {"instruction": "Test instruction 2", "input": ""}
        formatted = format_input(entry)
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction 2")
        self.assertEqual(formatted, expected)

    def test_dataset_initialization(self):
        dataset = InstructionDataset(self.test_data, self.tokenizer)
        self.assertEqual(len(dataset), len(self.test_data))

    def test_dataset_getitem(self):
        dataset = InstructionDataset(self.test_data, self.tokenizer)

        # Test getting the first sample
        sample = dataset[0]
        self.assertIsInstance(sample, list)

        # Convert to tensor for verification
        sample_tensor = torch.tensor(sample)
        self.assertEqual(sample_tensor.ndim, 1)

    def test_collate_fn(self):
        dataset = InstructionDataset(self.test_data, self.tokenizer)
        batch = [dataset[0], dataset[1]]

        inputs, targets = InstructionDataset.collate_fn(batch,
                                                        pad_token_id=50256)

        # Check shapes and types
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(inputs.shape[0], 2)  # Batch size
        self.assertEqual(targets.shape[0], 2)  # Batch size
        self.assertEqual(inputs.shape, targets.shape)

    def test_dataset_tokenization(self):
        dataset = InstructionDataset(self.test_data, self.tokenizer)

        # Get raw text from the first sample
        entry = self.test_data[0]

        # Get the first sample from the dataset
        sample = dataset[0]

        # Decode tokens and verify it contains parts of the formatted input
        decoded_text = self.tokenizer.decode(sample)

        # Check that key parts of the instruction are in the decoded text
        self.assertIn("Instruction", decoded_text)
        self.assertIn(entry["instruction"], decoded_text)
        self.assertIn(entry["output"], decoded_text)


if __name__ == "__main__":
    unittest.main()
