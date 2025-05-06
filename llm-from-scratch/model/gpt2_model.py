import math

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with causal masking.

    Implements scaled dot-product attention with multiple heads and causal masking
    to prevent positions from attending to subsequent positions.

    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension (must be divisible by num_heads)
        context_length (int): Maximum sequence length
        dropout (float): Dropout probability
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to include bias in query, key, value projections. Defaults to False.
    """

    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(
            d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_tokens, d_in]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_tokens, d_out]
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2,
                                               3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply scaling factor and softmax
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
        # Transpose: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    """Layer normalization module.

    Args:
        emb_dim (int): Embedding dimension to normalize
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Apply GELU activation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after GELU activation
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))))


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        cfg (dict): Configuration dictionary containing emb_dim
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
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
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"],
                                      d_out=cfg["emb_dim"],
                                      context_length=cfg["context_length"],
                                      num_heads=cfg["n_heads"],
                                      dropout=cfg["drop_rate"],
                                      qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

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
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
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

        self.final_norm = LayerNorm(cfg["emb_dim"])
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


def generate_text_greedy(model, idx, max_new_tokens, context_size):
    """Generate text using greedy decoding.

    This function generates text by always selecting the most probable next token
    at each step (greedy decoding).

    Args:
        model (GPTModel): The language model to use for generation
        idx (torch.Tensor): Starting token indices of shape [batch_size, seq_len]
        max_new_tokens (int): Number of new tokens to generate
        context_size (int): Maximum context length the model can process

    Returns:
        torch.Tensor: Generated token indices of shape [batch_size, seq_len + max_new_tokens]
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1),
                                             target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,
                                      model,
                                      device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,
                                    model,
                                    device,
                                    num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_with_temperature_and_topk(model,
                                            idx,
                                            max_new_tokens,
                                            context_size,
                                            temperature=0.0,
                                            top_k=None,
                                            eos_id=None):
    """Generate text using various sampling strategies.
    This function generates text by either:
    1. Greedy decoding (temperature=0.0): Always selecting the most probable next token
    2. Temperature sampling (temperature>0.0): Sampling from a softmax distribution with temperature
    3. Top-k sampling (top_k is not None): Limiting sampling to only the top-k most likely tokens

    Args:
        model (GPTModel): The language model to use for generation
        idx (torch.Tensor): Starting token indices of shape [batch_size, seq_len]
        max_new_tokens (int): Number of new tokens to generate
        context_size (int): Maximum context length the model can process
        temperature (float, optional): Controls randomness in sampling. Higher values produce more
                                      diverse outputs. If 0.0, uses greedy decoding. Defaults to 0.0.
        top_k (int, optional): If specified, limits sampling to the top-k most likely tokens. Defaults to None.
        eos_id (int, optional): Token ID that signals end of sequence. If generated, stops early. Defaults to None.

    Returns:
        torch.Tensor: Generated token indices of shape [batch_size, seq_len + max_new_tokens]
                     (or shorter if eos_id is generated)
    """
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Crop context to maximum supported length
        idx_cond = idx[:, -context_size:]

        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last token's predictions
        logits = logits[:, -1, :]

        # Apply top-k filtering if specified
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            # Get minimum value from top-k
            min_val = top_logits[:, -1]
            # Set all logits below min_val to -infinity
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)

        # Apply temperature sampling if temperature > 0
        if temperature > 0.0:
            # Scale logits by temperature (lower temperature = more deterministic)
            logits = logits / temperature

            # Convert to probability distribution
            probs = torch.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

        # Otherwise use greedy decoding (select highest probability token)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop generation if end-of-sequence token is encountered
        if eos_id is not None and (idx_next == eos_id).any():
            if (idx_next == eos_id).all():
                break

        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_sample_text_greedy(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_greedy(model=model,
                                         idx=encoded,
                                         max_new_tokens=50,
                                         context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    model.train()
    return decoded_text.replace("\n", " ")


def generate_sample_text_with_temperature_and_topk(model,
                                                   tokenizer,
                                                   device,
                                                   start_context,
                                                   max_new_tokens=50,
                                                   temperature=0.8,
                                                   top_k=40,
                                                   eos_id=None):
    """Generate text using temperature and top-k sampling.

    This function generates text by sampling from the probability distribution of next tokens,
    controlled by temperature and limited to the top-k most likely tokens.

    Args:
        model (GPTModel): The language model to use for generation
        tokenizer: Tokenizer for encoding/decoding text
        device: Device to run generation on (cpu or cuda)
        start_context (str): Starting text prompt
        max_new_tokens (int): Number of new tokens to generate
        temperature (float): Controls randomness (higher = more random)
        top_k (int): Limits sampling to the top k most likely tokens
        eos_id (int, optional): Token ID that signals end of sequence

    Returns:
        str: Generated text including the start context
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_with_temperature_and_topk(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id)
        decoded_text = token_ids_to_text(token_ids, tokenizer)

    model.train()
    return decoded_text.replace("\n", " ")


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: dict):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"],
                                 3,
                                 axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"],
                                 3,
                                 axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
