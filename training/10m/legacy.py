"""Historical version-1 character GPT, retained for checkpoint compatibility only.

This is the old LayerNorm/learned-position/GELU implementation, not a source of
building blocks for the current architecture. The complete modern implementation
and configuration live in model.py. Preserve this code's historical behavior.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


# Model: B = batch, T = sequence length, C = embedding width, H = heads.
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    def __post_init__(self):
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("all model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")


def causal_attention(query, key, value):
    # Q, K, V: [B, H, T, C/H]. A row may see itself and earlier columns only.
    # For each query position, compare its feature vector with every key vector.
    # Q and K are [B, H, T, D], so transpose K's last two axes to [B, H, D, T].
    # This makes the matrix product [B, H, T, D] @ [B, H, D, T] yield
    # [B, H, T, T]: one score for each query/key position pair. K itself is
    # unchanged; transpose only changes the view of its axes for this product.
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    # Example causal mask for T = 4 (rows are queries; columns are keys):
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    mask = torch.ones(query.size(-2), key.size(-2), device=query.device, dtype=torch.bool).tril()
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return weights @ value


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(
            3, dim=-1
        )  # Split the combined projection into its three equal parts: query, key, and value.
        # Split the embedding width evenly across heads; each head processes width / n_head features.
        query, key, value = [
            tensor.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        attended = causal_attention(query, key, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(attended)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.n_embd
        )  # n_embd is the embedding width (number of features per token).
        self.attention = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.lm_head(
            self.final_norm(self.blocks(x))
        )  # [B, T, vocabulary]; normalize the final hidden states before projecting them to token scores.

        return logits
