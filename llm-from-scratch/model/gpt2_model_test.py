import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from model.gpt2_model import (GELU, FeedForwardNetwork, GPTModel, LayerNorm,
                              MultiHeadAttention, TransformerBlock,
                              generate_text_greedy)


class TestGPT2Model(unittest.TestCase):

    def setUp(self):
        # Common configuration for tests
        self.model_config = {
            "vocab_size": 100,
            "emb_dim": 64,
            "context_length": 16,
            "n_heads": 4,
            "n_layers": 2,
            "drop_rate": 0.1,
            "qkv_bias": True
        }

    def test_multi_head_attention(self):
        # Test initialization and forward pass
        batch_size, seq_len, d_in = 2, 8, 64
        d_out, num_heads = 64, 4

        mha = MultiHeadAttention(d_in=d_in,
                                 d_out=d_out,
                                 context_length=16,
                                 dropout=0.1,
                                 num_heads=num_heads)

        x = torch.randn(batch_size, seq_len, d_in)
        output = mha(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_out))

    def test_layer_norm(self):
        batch_size, seq_len, emb_dim = 2, 8, 64
        ln = LayerNorm(emb_dim)

        x = torch.randn(batch_size, seq_len, emb_dim)
        output = ln(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Check that normalization happened along the last dimension
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        # Mean should be close to 0 and variance close to 1 before scaling and shifting
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean),
                                       atol=1e-5))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-5))

    def test_gelu(self):
        gelu = GELU()
        x = torch.randn(10)
        output = gelu(x)

        # GELU should be bounded and monotonically increasing
        self.assertEqual(output.shape, x.shape)

        # Test specific known values
        self.assertTrue(
            torch.isclose(gelu(torch.tensor(0.0)), torch.tensor(0.0)))
        self.assertTrue(
            torch.isclose(gelu(torch.tensor(2.0)),
                          torch.tensor(1.9545),
                          atol=1e-4))
        self.assertTrue(
            torch.isclose(gelu(torch.tensor(-2.0)),
                          torch.tensor(-0.0455),
                          atol=1e-4))

    def test_feed_forward_network(self):
        batch_size, seq_len = 2, 8
        ffn = FeedForwardNetwork(self.model_config)

        x = torch.randn(batch_size, seq_len, self.model_config["emb_dim"])
        output = ffn(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

    def test_transformer_block(self):
        batch_size, seq_len = 2, 8
        block = TransformerBlock(self.model_config)

        x = torch.randn(batch_size, seq_len, self.model_config["emb_dim"])
        output = block(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

    def test_gpt_model(self):
        batch_size, seq_len = 2, 8
        model = GPTModel(self.model_config)

        # Input is token indices
        input_ids = torch.randint(0, self.model_config["vocab_size"],
                                  (batch_size, seq_len))
        output = model(input_ids)

        # Check output shape - should be (batch_size, seq_len, vocab_size)
        self.assertEqual(
            output.shape,
            (batch_size, seq_len, self.model_config["vocab_size"]))

    def test_generate_text_greedy(self):
        batch_size, seq_len = 1, 4
        model = GPTModel(self.model_config)

        # Start with a small sequence
        input_ids = torch.randint(0, self.model_config["vocab_size"],
                                  (batch_size, seq_len))

        # Generate 3 new tokens
        max_new_tokens = 3
        output_ids = generate_text_greedy(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.model_config["context_length"])

        # Check output shape - should have 3 more tokens than input
        self.assertEqual(output_ids.shape,
                         (batch_size, seq_len + max_new_tokens))

        # Check that the beginning matches the input
        self.assertTrue(torch.all(output_ids[:, :seq_len] == input_ids))

    def test_model_integration(self):
        """Test the full model pipeline with a small example."""
        config = {
            "vocab_size": 10,
            "emb_dim": 32,
            "context_length": 8,
            "n_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False
        }

        model = GPTModel(config)

        # Set model to eval mode to disable dropout
        model.eval()

        # Create a simple input sequence
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # Get logits
        logits = model(input_ids)

        # Generate text
        generated = generate_text_greedy(model,
                                         input_ids,
                                         max_new_tokens=4,
                                         context_size=config["context_length"])

        # Basic checks
        self.assertEqual(logits.shape, (1, 4, config["vocab_size"]))
        self.assertEqual(generated.shape, (1, 8))  # original 4 + 4 new tokens


if __name__ == '__main__':
    unittest.main()
