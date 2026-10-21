"""10M decoder: RMSNorm, RoPE, SwiGLU and tied embeddings by default.

Keep 6 layers, width 384, 6 MHA heads, context 256, projection biases and
vanilla weight initialization. architecture_version=1 reproduces the historical
LayerNorm/learned-position/GELU model only for legacy checkpoint compatibility.
"""

from dataclasses import asdict, dataclass

import torch
from model_10m import legacy
from torch import nn
from torch.nn import functional as F

__all__ = ["GPT", "Attention", "Block", "ModelConfig", "RMSNorm", "SwiGLU", "causal_attention"]
MODEL_DEPENDENCIES = {"legacy_definition": legacy.__file__}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    architecture_version: int = 2

    def __post_init__(self):
        if type(self.architecture_version) is not int or self.architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 (legacy) or 2 (RMSNorm/RoPE/SwiGLU)")
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("all model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.architecture_version == 2 and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")


def causal_attention(query, key, value):
    """Standard causal SDPA; CUDA selects FlashAttention where supported.

    Same scale, mask and zero dropout as the explicit legacy reference. The legacy
    architecture retains its original implementation in model_10m.legacy.
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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)

    def cache_rotary(self, length):
        width = self.qkv.in_features // self.n_head
        frequencies = 10000.0 ** (
            -torch.arange(0, width, 2, device=self.qkv.weight.device).float() / width
        )
        angles = torch.outer(torch.arange(length, device=frequencies.device).float(), frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def rotate(self, x):
        if not hasattr(self, "rope_cos"):
            return rotary(x)
        cos = self.rope_cos[: x.size(-2)].to(x.dtype)
        sin = self.rope_sin[: x.size(-2)].to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, -1)
        query, key, value = [
            tensor.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        attended = causal_attention(self.rotate(query), self.rotate(key), value)
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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.architecture_version == 1:
            # Historical blocks only: the modern architecture never inherits them.
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            block_type, norm_type = legacy.Block, nn.LayerNorm
        else:
            block_type, norm_type = Block, RMSNorm
        self.blocks = nn.Sequential(*(block_type(config) for _ in range(config.n_layer)))
        self.final_norm = norm_type(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)
        if config.architecture_version == 2:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, labels=None, loss_chunk_size=0):
        if not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        x = self.token_embedding(tokens)
        if self.config.architecture_version == 1:
            positions = torch.arange(tokens.size(1), device=tokens.device)
            x = x + self.position_embedding(positions)
        hidden = self.final_norm(self.blocks(x))
        if labels is None or self.config.architecture_version == 1:
            return self.lm_head(hidden)
        if not loss_chunk_size:
            return F.cross_entropy(
                self.lm_head(hidden).flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        # Recompute each projection during backward, bounding vocabulary-logit memory.
        from torch.utils.checkpoint import checkpoint

        def chunk_loss(features, targets):
            return F.cross_entropy(
                self.lm_head(features), targets, ignore_index=-100, reduction="sum"
            )

        return sum(
            checkpoint(chunk_loss, features, targets, use_reentrant=False)
            for features, targets in zip(
                hidden.flatten(0, 1).split(loss_chunk_size), labels.flatten().split(loss_chunk_size)
            )
        )
