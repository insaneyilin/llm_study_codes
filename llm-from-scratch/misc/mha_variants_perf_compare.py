import math
import timeit
from statistics import mean, stdev

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Setup
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"PyTorch version: {torch.__version__}")
print(f"Running on {device}")

# Parameters
batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# Define all MHA implementations


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(  # `_` ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class Ch03_MHA_Wrapper(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


class Ch03_MHA(nn.Module):

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
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(
            d_out, d_out)  # optional for combining the outputs of each head
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split into multiple heads
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Self attention scores for each head
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Causal mask, we can only see the current and previous tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Mask invalid positions with -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # merge heads, self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # Optional projection

        return context_vec


# MultiHeadAttentionCombinedQKV use only one linear layer for Q, K, V rather than three linear layers for Q, K, V
class MultiHeadAttentionCombinedQKV(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 context_length,
                 dropout=0.0,
                 qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        # Merge W_Q, W_K, W_V into one linear layer
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads,
                       self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5,
                                     dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens,
                                                    embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec


class MHAEinsum(nn.Module):

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
        self.head_dim = d_out // num_heads

        # Initialize parameters for Q, K, V
        self.W_query = nn.Parameter(torch.randn(d_out, d_in))
        self.W_key = nn.Parameter(torch.randn(d_out, d_in))
        self.W_value = nn.Parameter(torch.randn(d_out, d_in))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x):
        b, n, _ = x.shape

        # Calculate Q, K, V using einsum, first perform linear transformations
        # (b, n, d_in) @ (d_in, d_out) --> (b, n, d_out)
        Q = torch.einsum("bnd,di->bni", x, self.W_query)
        K = torch.einsum("bnd,di->bni", x, self.W_key)
        V = torch.einsum("bnd,di->bni", x, self.W_value)

        # Add biases if they are used
        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        # Reshape for multi-head attention
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim**0.5)

        # Apply mask
        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(
            b, self.num_heads, n, n)
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate the attended context vectors
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # Combine heads and project the output
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 context_length,
                 dropout=0.0,
                 qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads,
                       self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


# SDPA(Scaled Dot Product Attention) without FlashAttention
class MHAPyTorchSDPAWithoutFlash(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 context_length,
                 dropout=0.0,
                 qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads,
                       self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        context_vec = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=use_dropout,
            is_causal=False)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


class MHAPyTorchClass(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 context_length,
                 dropout=0.0,
                 qkv_bias=False,
                 need_weights=True):
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        # If need_weights is False, we don't need `attn_output_weights`(attn scores)
        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(x,
                                             x,
                                             x,
                                             attn_mask=attn_mask,
                                             need_weights=self.need_weights)

        output = self.proj(attn_output)

        return output


# Initialize all MHA variants
mha_implementations = {
    "1) MHA wrapper class":
    Ch03_MHA_Wrapper(d_in=embed_dim,
                     d_out=embed_dim // 12,
                     context_length=context_len,
                     dropout=0.0,
                     num_heads=12,
                     qkv_bias=False).to(device),
    "2) MHA Ch03":
    Ch03_MHA(d_in=embed_dim,
             d_out=embed_dim,
             context_length=context_len,
             dropout=0.0,
             num_heads=12,
             qkv_bias=False).to(device),
    "3) MHA with combined QKV":
    MultiHeadAttentionCombinedQKV(d_in=embed_dim,
                                  d_out=embed_dim,
                                  context_length=context_len,
                                  dropout=0.0,
                                  num_heads=12,
                                  qkv_bias=False).to(device),
    "4) MHA with Einsum":
    MHAEinsum(d_in=embed_dim,
              d_out=embed_dim,
              context_length=context_len,
              dropout=0.0,
              num_heads=12,
              qkv_bias=False).to(device),
    "5) PyTorch SDPA":
    MHAPyTorchScaledDotProduct(d_in=embed_dim,
                               d_out=embed_dim,
                               context_length=context_len,
                               dropout=0.0,
                               num_heads=12,
                               qkv_bias=False).to(device),
    "6) PyTorch SDPA no Flash":
    MHAPyTorchSDPAWithoutFlash(d_in=embed_dim,
                               d_out=embed_dim,
                               context_length=context_len,
                               dropout=0.0,
                               num_heads=12,
                               qkv_bias=False).to(device),
    "7) PyTorch MHA class":
    MHAPyTorchClass(d_in=embed_dim,
                    d_out=embed_dim,
                    context_length=context_len,
                    dropout=0.0,
                    num_heads=12,
                    qkv_bias=False).to(device),
    "8) PyTorch MHA no weights":
    MHAPyTorchClass(d_in=embed_dim,
                    d_out=embed_dim,
                    context_length=context_len,
                    dropout=0.0,
                    num_heads=12,
                    qkv_bias=False,
                    need_weights=False).to(device)
}


# Performance measurement
def measure_performance(implementation, input_data, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = implementation(input_data)

    # Measure
    times = []
    for _ in range(num_runs):
        start_time = timeit.default_timer()
        _ = implementation(input_data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    return mean(times), stdev(times)


# Compare performance
results = {}
for name, impl in mha_implementations.items():
    mean_time, std_time = measure_performance(impl, embeddings)
    results[name] = (mean_time, std_time)
    print(f"{name}: {mean_time:.2f} ± {std_time:.2f} ms")


# Visualization
def plot_results(results, filename="mha_performance_comparison.png"):
    names = list(results.keys())
    means = [val[0] for val in results.values()]
    stds = [val[1] for val in results.values()]

    # Customize for dark mode aesthetics
    plt.rcParams["figure.facecolor"] = "#121212"
    plt.rcParams["axes.facecolor"] = "#121212"
    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["grid.color"] = "#444444"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 8

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names,
                  means,
                  yerr=stds,
                  capsize=5,
                  color='skyblue',
                  error_kw={
                      'ecolor': 'grey',
                      'elinewidth': 2
                  })

    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title("Multi-Head Attention Implementation Performance",
                 fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + 0.5,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


plot_results(results)

# PyTorch version: 2.7.0
# Running on cpu
# 1) MHA wrapper class: 164.66 ± 12.25 ms
# 2) MHA Ch03: 172.28 ± 25.29 ms
# 3) MHA with combined QKV: 176.17 ± 10.48 ms
# 4) MHA with Einsum: 175.25 ± 4.95 ms
# 5) PyTorch SDPA: 60.67 ± 0.96 ms
# 6) PyTorch SDPA no Flash: 67.93 ± 1.22 ms
# 7) PyTorch MHA class: 175.46 ± 4.60 ms
# 8) PyTorch MHA no weights: 88.61 ± 1.09 ms

# PyTorch version: 2.7.0
# Running on mps
# 1) MHA wrapper class: 40.73 ± 1.51 ms
# 2) MHA Ch03: 38.36 ± 16.23 ms
# 3) MHA with combined QKV: 43.89 ± 1.46 ms
# 4) MHA with Einsum: 50.44 ± 1.61 ms
# 5) PyTorch SDPA: 56.05 ± 0.37 ms
# 6) PyTorch SDPA no Flash: 56.47 ± 1.98 ms
# 7) PyTorch MHA class: 41.24 ± 8.63 ms
# 8) PyTorch MHA no weights: 61.52 ± 3.25 ms
