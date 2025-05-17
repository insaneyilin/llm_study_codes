import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device_name = "cpu"
if torch.backends.mps.is_available():
    device_name = "mps"
elif torch.cuda.is_available():
    device_name = "cuda"
device = torch.device(device_name)
print(f"Using device: {device}")


# 定义基础的单头注意力
class CausalAttention(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout: float,
                 qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens],
                                 -torch.inf)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5),
                                     dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


# 实现1：多头注意力的Wrapper版本
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout: float,
                 num_heads: int,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out // num_heads, context_length, dropout,
                            qkv_bias) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(
            torch.cat([head(x) for head in self.heads], dim=-1))


# 实现2：高效的多头注意力
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout: float,
                 num_heads: int,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads,
                                       self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads,
                                      self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens],
                                 -torch.inf)
        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5),
                                     dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


# 基准测试函数
def benchmark(model: nn.Module,
              x: torch.Tensor,
              warmup: int = 10,
              repeats: int = 100) -> Tuple[float, float]:
    """返回平均时间和标准差(毫秒)"""
    # Warmup
    for _ in range(warmup):
        _ = model(x)

    # 正式测试
    times = []
    if device_name == "cuda":
        torch.cuda.synchronize()
    for _ in range(repeats):
        start = time.time()
        _ = model(x)
        if device_name == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)  # 转换为毫秒

    return float(torch.tensor(times).mean()), float(torch.tensor(times).std())


# 测试配置
configs = [
    {
        "batch_size": 128,
        "context_length": 512,
        "d_in": 256,
        "d_out": 512,
        "num_heads": 4
    },
    {
        "batch_size": 128,
        "context_length": 512,
        "d_in": 256,
        "d_out": 512,
        "num_heads": 8
    },
]

# 运行测试
results = []
for cfg in configs:
    # 生成测试数据
    torch.manual_seed(42)
    x = torch.randn(cfg["batch_size"], cfg["context_length"],
                    cfg["d_in"]).to(device)

    wrapper = MultiHeadAttentionWrapper(cfg["d_in"], cfg["d_out"],
                                        cfg["context_length"], 0.1,
                                        cfg["num_heads"]).to(device)

    efficient = MultiHeadAttention(cfg["d_in"], cfg["d_out"],
                                   cfg["context_length"], 0.1,
                                   cfg["num_heads"]).to(device)

    # 基准测试
    t_wrapper, std_wrapper = benchmark(wrapper, x)
    t_efficient, std_efficient = benchmark(efficient, x)

    results.append({
        "config": cfg,
        "wrapper_time": t_wrapper,
        "wrapper_std": std_wrapper,
        "efficient_time": t_efficient,
        "efficient_std": std_efficient,
        "speedup": t_wrapper / t_efficient
    })

# 打印结果
print("\nBenchmark Results:")
for res in results:
    cfg = res["config"]
    print(
        f"\nConfig: bs={cfg['batch_size']}, seq={cfg['context_length']}, "
        f"d_in={cfg['d_in']}, d_out={cfg['d_out']}, heads={cfg['num_heads']}")
    print(f"Wrapper: {res['wrapper_time']:.2f} ± {res['wrapper_std']:.2f} ms")
    print(
        f"Efficient: {res['efficient_time']:.2f} ± {res['efficient_std']:.2f} ms"
    )
    print(f"Speedup: {res['speedup']:.2f}x")

# 可视化
labels = [f"Config {i+1}" for i in range(len(results))]
wrapper_times = [res["wrapper_time"] for res in results]
efficient_times = [res["efficient_time"] for res in results]

plt.figure(figsize=(10, 6))
x = range(len(labels))
width = 0.35
plt.bar(x,
        wrapper_times,
        width,
        label='Wrapper Implementation',
        yerr=[res["wrapper_std"] for res in results])
plt.bar([p + width for p in x],
        efficient_times,
        width,
        label='Efficient Implementation',
        yerr=[res["efficient_std"] for res in results])
plt.ylabel('Inference Time (ms)')
plt.title('Performance Comparison of MHA Implementations')
plt.xticks([p + width / 2 for p in x], labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('mha_performance_comparison.png')
plt.show()

# Benchmark Results:

# Config: bs=128, seq=512, d_in=256, d_out=512, heads=4
# Wrapper: 116.23 ± 4.59 ms
# Efficient: 119.27 ± 9.11 ms
# Speedup: 0.97x

# Config: bs=128, seq=512, d_in=256, d_out=512, heads=8
# Wrapper: 202.17 ± 11.30 ms
# Efficient: 180.74 ± 1.73 ms
# Speedup: 1.12x
