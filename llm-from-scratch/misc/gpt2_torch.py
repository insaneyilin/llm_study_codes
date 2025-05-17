import math

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        cfg (dict): Configuration dictionary containing emb_dim
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """Apply feed-forward transformation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network.

    Args:
        cfg (dict): Configuration dictionary containing model parameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=cfg["emb_dim"],
                                         num_heads=cfg["n_heads"],
                                         dropout=cfg["drop_rate"],
                                         bias=cfg["qkv_bias"],
                                         batch_first=True)
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
        self.context_length = cfg["context_length"]

        # Register causal mask as a buffer
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(cfg["context_length"],
                                  cfg["context_length"]),
                       diagonal=1).bool())

    def forward(self, x):
        """Apply transformer block operations.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # Get the actual sequence length (might be less than context_length)
        seq_len = x.size(1)
        attn_mask = self.mask[:seq_len, :seq_len]

        x, _ = self.att(query=x,
                        key=x,
                        value=x,
                        attn_mask=attn_mask,
                        need_weights=False)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """GPT-style autoregressive language model.

    Args:
        cfg (dict): Configuration dictionary containing model parameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"],
                                  cfg["vocab_size"],
                                  bias=False)

    def forward(self, in_idx):
        """Forward pass through the GPT model.

        Args:
            in_idx (torch.Tensor): Input token indices of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
