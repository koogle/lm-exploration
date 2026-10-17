"""100M-family decoder: RMSNorm, RoPE, SwiGLU and tied embeddings by default.

Keep 14 layers, width 768, 12 MHA heads, context 256, projection biases and
vanilla weight initialization. architecture_version=1 reproduces the historical
LayerNorm/learned-position/GELU model only for legacy checkpoint compatibility.
"""

from dataclasses import dataclass

import torch
from model_10m import model as vanilla
from torch import nn
from torch.nn import functional as F

__all__ = ["GPT", "Attention", "Block", "ModelConfig", "RMSNorm", "SwiGLU", "causal_attention"]
MODEL_DEPENDENCIES = {"model_10m_definition": vanilla.__file__}


@dataclass(frozen=True)
class ModelConfig(vanilla.ModelConfig):
    n_layer: int = 14
    n_head: int = 12
    n_embd: int = 768
    architecture_version: int = 2

    def __post_init__(self):
        if type(self.architecture_version) is not int or self.architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 (legacy) or 2 (RMSNorm/RoPE/SwiGLU)")
        super().__post_init__()
        if self.architecture_version == 2 and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")


def causal_attention(query, key, value):
    """Standard causal SDPA; CUDA selects FlashAttention where supported.

    Same scale, mask and zero dropout as the explicit 10M reference. The legacy
    architecture retains its original implementation in model_10m.
    """
    return F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps  # Match the historical LayerNorm epsilon.

    def forward(self, x):
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotary(x, offset=0):
    """Rotate adjacent query/key feature pairs using position and theta 10,000."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequencies = 10000.0 ** (
        -torch.arange(0, x.size(-1), 2, device=x.device, dtype=dtype) / x.size(-1)
    )
    positions = torch.arange(offset, offset + x.size(-2), device=x.device, dtype=dtype)
    angles = torch.outer(positions, frequencies)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(vanilla.Attention):
    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, -1)
        query, key, value = [
            tensor.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        attended = causal_attention(rotary(query), rotary(key), value)
        return self.projection(attended.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden_width = (8 * width + 2) // 3
        self.gate = nn.Linear(width, self.hidden_width)
        self.up = nn.Linear(width, self.hidden_width)
        self.down = nn.Linear(self.hidden_width, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(vanilla.Block):
    def __init__(self, config):
        super().__init__(config)
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd)


class GPT(vanilla.GPT):
    def __init__(self, config):
        if config.architecture_version == 1:
            super().__init__(config)
            return
        nn.Module.__init__(self)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens):
        if self.config.architecture_version == 1:
            return super().forward(tokens)
        if not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        return self.lm_head(self.final_norm(self.blocks(self.token_embedding(tokens))))
