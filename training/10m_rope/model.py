"""10M RoPE ablation: replace learned positions; retain LayerNorm, GELU and MHA.

RoPE rotates adjacent pairs of query/key features in each layer, theta 10,000.
Keep the original explicit causal-attention implementation, biases and weight
initialization. Context remains bounded by block_size; this adds no KV cache.
"""

from dataclasses import dataclass

import torch
from model_10m import model as vanilla
from torch import nn

BASE_MODEL_SOURCE = vanilla.__file__


@dataclass(frozen=True)
class ModelConfig(vanilla.ModelConfig):
    def __post_init__(self):
        super().__post_init__()
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")


def rotary(x, offset=0):
    """Rotate Q/K [batch, heads, time, head_dim], using absolute token positions."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequencies = 10000.0 ** (
        -torch.arange(0, x.size(-1), 2, device=x.device, dtype=dtype) / x.size(-1)
    )
    positions = torch.arange(offset, offset + x.size(-2), device=x.device, dtype=dtype)
    angles = torch.outer(positions, frequencies)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(vanilla.Attention):
    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query, key, value = [
            tensor.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        attended = vanilla.causal_attention(rotary(query), rotary(key), value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(attended)


class Block(vanilla.Block):
    def __init__(self, config):
        super().__init__(config)
        self.attention = Attention(config)


class GPT(vanilla.GPT):
    def __init__(self, config):
        # Build the vanilla structure without its learned positional embedding.
        nn.Module.__init__(self)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)

    def forward(self, tokens):
        if not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        x = self.token_embedding(tokens)
        return self.lm_head(self.final_norm(self.blocks(x)))
