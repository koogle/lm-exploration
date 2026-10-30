"""10M decoder: RMSNorm, RoPE, SwiGLU and tied embeddings by default.

Keep 6 layers, width 384, 6 MHA heads, context 256, projection biases and
vanilla weight initialization.

Every forward pass is masked by document: packed documents end in EOS
(`ModelConfig.eos_id`), a token attends only to earlier tokens of its own
document, and RoPE positions restart after each EOS.
"""

from dataclasses import KW_ONLY, asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from trainer.roles import declare

__all__ = [
    "GPT",
    "Attention",
    "Block",
    "Documents",
    "ModelConfig",
    "RMSNorm",
    "SwiGLU",
    "document_attention",
    "find_documents",
    "rotary",
]
MODEL_DEPENDENCIES = {}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    _: KW_ONLY
    eos_id: int

    def __post_init__(self):
        dimensions = {k: v for k, v in asdict(self).items() if k != "eos_id"}
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("all model dimensions must be positive")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError("eos_id must be a token id")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")


class Documents(NamedTuple):
    """Packed-document layout of a [B, T] token batch."""

    ids: torch.Tensor  # [B, T], restarting at 0 in each row
    positions: torch.Tensor  # [B, T], position within the token's document
    cu_seqlens: torch.Tensor  # int32 start of every document in the flattened batch, then B * T


@torch.compiler.disable
def find_documents(tokens, eos_id):
    """A new document starts at each row start and after every EOS input token."""
    batch, length = tokens.shape
    start = torch.ones_like(tokens, dtype=torch.bool)
    start[:, 1:] = tokens[:, :-1] == eos_id
    index = torch.arange(length, device=tokens.device).expand_as(tokens)
    positions = index - torch.where(start, index, 0).cummax(1).values
    starts = start.flatten().nonzero().squeeze(1)
    end = torch.tensor([batch * length], device=tokens.device)
    return Documents(
        start.long().cumsum(1) - 1, positions, torch.cat((starts, end)).to(torch.int32)
    )


def document_attention(query, key, value, documents):
    """Causal attention within each document; q/k/v and the output are [B, T, H, D].

    bf16/fp16 CUDA tensors run FlashAttention's varlen kernel over the flattened
    batch; otherwise SDPA with an explicit document-causal mask. Scale 1/sqrt(D).
    """
    batch, length, heads, width = query.shape
    if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
        from torch.nn.attention.varlen import varlen_attn

        flat = [t.reshape(batch * length, heads, width) for t in (query, key, value)]
        cu = documents.cu_seqlens
        attended = varlen_attn(*flat, cu, cu, length, length, window_size=(-1, 0))
        return attended.view(batch, length, heads, width)
    ids = documents.ids
    causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
    allowed = (ids[:, :, None] == ids[:, None, :]) & causal
    attended = F.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (query, key, value)), attn_mask=allowed[:, None]
    )
    return attended.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotary(x, positions):
    """Rotate adjacent feature pairs of x by `positions` (broadcast to x.shape[:-1]), theta 10,000."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequencies = 10000.0 ** (
        -torch.arange(0, x.size(-1), 2, device=x.device, dtype=dtype) / x.size(-1)
    )
    angles = positions.to(dtype)[..., None] * frequencies
    return rotate_pairs(x, angles.cos().to(x.dtype), angles.sin().to(x.dtype))


def rotate_pairs(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)

    def optimizer(self):
        yield from declare("muon", self.qkv.weight, self.projection.weight)
        yield from declare("adamw_decay", self.qkv.bias, self.projection.bias)

    def cache_rotary(self, length):
        width = self.qkv.in_features // self.n_head
        frequencies = 10000.0 ** (
            -torch.arange(0, width, 2, device=self.qkv.weight.device).float() / width
        )
        angles = torch.outer(torch.arange(length, device=frequencies.device).float(), frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def rotate(self, x, positions):
        """x is [B, T, H, D]; positions [B, T]."""
        positions = positions[:, :, None]
        if not hasattr(self, "rope_cos"):
            return rotary(x, positions)
        return rotate_pairs(
            x, self.rope_cos[positions].to(x.dtype), self.rope_sin[positions].to(x.dtype)
        )

    def forward(self, x, documents):
        batch, length, width = x.shape
        query, key, value = [
            tensor.view(batch, length, self.n_head, width // self.n_head)
            for tensor in self.qkv(x).chunk(3, -1)
        ]
        positions = documents.positions
        attended = document_attention(
            self.rotate(query, positions), self.rotate(key, positions), value, documents
        )
        return self.projection(attended.reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden=None, bias=True):
        super().__init__()
        self.hidden_width = hidden or (8 * width + 2) // 3
        self.gate = nn.Linear(width, self.hidden_width, bias=bias)
        self.up = nn.Linear(width, self.hidden_width, bias=bias)
        self.down = nn.Linear(self.hidden_width, width, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd)

    def optimizer(self):
        # The block declares its shared children (RMSNorm, SwiGLU); attention its own.
        linears = (self.mlp.gate, self.mlp.up, self.mlp.down)
        yield from declare("muon", *(linear.weight for linear in linears))
        yield from declare(
            "adamw_decay",
            *(linear.bias for linear in linears if linear.bias is not None),
            self.attention_norm,
            self.mlp_norm,
        )

    def forward(self, x, documents):
        x = x + self.attention(self.attention_norm(x), documents)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)
        self.lm_head.weight = self.token_embedding.weight

    def optimizer(self):
        # Blocks declare their own parameters; the tied embedding/output table and the
        # final norm are not hidden matrices.
        yield from declare("adamw_decay", self.token_embedding, self.final_norm)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def trunk(self, tokens):
        """Residual stream after the last block, and the documents it was masked by."""
        if not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        documents = find_documents(tokens, self.config.eos_id)
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x, documents)
        return x, documents

    def forward(self, tokens, labels=None, loss_chunk_size=0, loss_fn=F.cross_entropy):
        """Logits, or the summed loss when `labels` are given.

        `loss_fn(logits, targets, ignore_index=..., reduction=...)` follows
        F.cross_entropy; the trainer passes `trainer.loss.cross_entropy`.
        """
        hidden = self.final_norm(self.trunk(tokens)[0])
        if labels is None:
            return self.lm_head(hidden)
        if not loss_chunk_size:
            return loss_fn(
                self.lm_head(hidden).flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        # Recompute each projection during backward, bounding vocabulary-logit memory.
        from torch.utils.checkpoint import checkpoint

        def chunk_loss(features, targets):
            return loss_fn(self.lm_head(features), targets, ignore_index=-100, reduction="sum")

        return sum(
            checkpoint(chunk_loss, features, targets, use_reentrant=False)
            for features, targets in zip(
                hidden.flatten(0, 1).split(loss_chunk_size), labels.flatten().split(loss_chunk_size)
            )
        )
