"""Hybrid-family decoder backbone.

Pre-norm blocks reuse RMSNorm (fp32 statistics) and SwiGLU from model_10m, the
SwiGLU without biases at a configurable width; tied embeddings and no biases.
Full-attention layers use GQA with head dimension 128, QK-norm and RoPE with a
configurable base. Every forward pass masks attention by document: documents
end in EOS (`HybridConfig.eos_id`) and a token sees only earlier tokens of its
own document. On CUDA attention runs our own kernel (document_attention.py),
bit-exact with the fixed-order definition in exact_attention.py. The loss is a chunked cross-entropy with a z-loss.
"""

import math
from dataclasses import asdict, dataclass

import torch
from model_10m.model import RMSNorm, SwiGLU
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .document_attention import document_attention, softmax_scale
from .documents import document_ids
from .exact_attention import document_starts

__all__ = [
    "Attention",
    "Block",
    "DocumentMask",
    "HybridConfig",
    "HybridLM",
    "RMSNorm",
    "SwiGLU",
    "chunked_cross_entropy",
]
IGNORE_INDEX = -100


class DocumentMask:
    """Document-aware causal mask: position q sees k <= q within its own document."""

    def __init__(self, documents):
        self.documents = documents
        # Documents are contiguous, so row i may attend to keys [start_i, i].
        self.starts = document_starts(documents)

    def dense(self):
        """[B, 1, T, T] boolean matrix of allowed attention."""
        length = self.documents.size(1)
        index = torch.arange(length, device=self.documents.device)
        same = self.documents[:, :, None] == self.documents[:, None, :]
        return (same & (index[None, :] <= index[:, None]))[:, None]


def reference_attention(query, key, value, allowed):
    """Explicit masked softmax attention with GQA, computed in at least fp32.

    Returns the output and the fp32 (or fp64) log-sum-exp of the scaled scores.
    """
    groups = query.size(1) // key.size(1)
    key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = query.to(dtype) @ key.to(dtype).transpose(-1, -2) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(~allowed, -math.inf)
    lse = scores.logsumexp(-1)
    weights = (scores - lse[..., None]).exp()
    return (weights @ value.to(dtype)).to(query.dtype), lse


def attend(query, key, value, mask):
    if query.is_cuda:
        return document_attention(query, key, value, mask.starts, softmax_scale(query.size(-1)))[0]
    return reference_attention(query, key, value, mask.dense())[0]


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int
    eos_id: int
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 1536
    n_head: int = 12
    n_kv_head: int = 4
    head_dim: int = 128
    mlp_hidden: int = 4096
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    z_loss: float = 1e-4
    loss_chunk_size: int = 8192

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in ("z_loss", "eos_id") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.z_loss < 0:
            raise ValueError("z_loss must be non-negative")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError("eos_id must be a token id")


def rope_tables(length, head_dim, base, device):
    frequencies = base ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim
    )
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float64), frequencies)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x, cos, sin):
    """Rotate halves of the head dimension; x is [B, H, T, D], tables [T or B, 1, T, D/2]."""
    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    first, second = values.chunk(2, dim=-1)
    cos, sin = cos.to(values.dtype), sin.to(values.dtype)
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return rotated.to(x.dtype)


def document_positions(documents):
    """Position of each token within its own document, [B, T]."""
    index = torch.arange(documents.size(1), device=documents.device).expand_as(documents)
    start = torch.zeros_like(documents, dtype=torch.bool)
    start[:, 0] = True
    start[:, 1:] = documents[:, 1:] != documents[:, :-1]
    return index - torch.where(start, index, 0).cummax(1).values


class Attention(nn.Module):
    """Document-causal GQA with QK-norm and RoPE."""

    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = (
            config.n_head,
            config.n_kv_head,
            config.head_dim,
        )
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        self.o_proj.residual_output = True

    def project(self, x, rope):
        batch, length, _ = x.shape
        query = self.q_proj(x).view(batch, length, self.n_head, self.head_dim)
        key = self.k_proj(x).view(batch, length, self.n_kv_head, self.head_dim)
        value = self.v_proj(x).view(batch, length, self.n_kv_head, self.head_dim)
        query, key = self.q_norm(query), self.k_norm(key)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        cos, sin = rope
        return apply_rope(query, cos, sin), apply_rope(key, cos, sin), value

    def forward(self, x, rope, mask):
        query, key, value = self.project(x, rope)
        attended = attend(query, key, value, mask)
        batch, length, _ = x.shape
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mixer_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mixer = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config.n_embd, config.mlp_hidden, bias=False)
        self.mlp.down.residual_output = True

    def forward(self, x, rope, mask):
        x = x + self.mixer(self.mixer_norm(x), rope, mask)
        return x + self.mlp(self.mlp_norm(x))


def _chunk_loss(features, weight, targets):
    logits = F.linear(features, weight)
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    lse = torch.logsumexp(logits, dim=-1)
    valid = targets != IGNORE_INDEX
    target_logits = logits.gather(-1, targets.clamp_min(0)[:, None]).squeeze(-1)
    ce = torch.where(valid, lse - target_logits, 0.0).sum()
    z = torch.where(valid, lse.square(), 0.0).sum()
    return ce, z


def chunked_cross_entropy(hidden, weight, labels, chunk_size):
    """Summed cross-entropy and summed squared log-normalizer over valid targets.

    Each chunk's logits are recomputed in backward, so peak memory holds one
    [chunk_size, vocab] fp32 block rather than the full logit tensor.
    """
    ce_sum = z_sum = 0.0
    for features, targets in zip(
        hidden.flatten(0, -2).split(chunk_size), labels.flatten().split(chunk_size)
    ):
        ce, z = checkpoint(_chunk_loss, features, weight, targets, use_reentrant=False)
        ce_sum, z_sum = ce_sum + ce, z_sum + z
    return ce_sum, z_sum


class HybridLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.apply(self.initialize)
        cos, sin = rope_tables(config.block_size, config.head_dim, config.rope_base, None)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def output_weight(self):
        return self.token_embedding.weight

    def initialize(self, module):
        """Normal(0, 0.02); residual output projections scaled by 1/sqrt(2 * n_layer)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.init_std
            if getattr(module, "residual_output", False):
                std /= math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, std=std)

    def hidden_states(self, tokens):
        """Final normalized hidden states.

        Each EOS ends a document: attention and RoPE positions restart after it.
        """
        length = tokens.size(1)
        if not 0 < length <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        documents = document_ids(tokens, self.config.eos_id)
        positions = document_positions(documents)
        rope = (self.rope_cos[positions][:, None], self.rope_sin[positions][:, None])
        mask = DocumentMask(documents)
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x, rope, mask)
        return self.final_norm(x)

    def forward(self, tokens, labels=None):
        """Logits, or summed losses when `labels` are given.

        Returns {"loss", "ce", "z", "count"}: `loss = ce + z_loss * z` summed over
        valid targets; divide by the global valid-target count to get the mean.
        """
        hidden = self.hidden_states(tokens)
        if labels is None:
            return F.linear(hidden, self.output_weight)
        ce, z = chunked_cross_entropy(
            hidden, self.output_weight, labels, self.config.loss_chunk_size
        )
        return {
            "loss": ce + self.config.z_loss * z,
            "ce": ce,
            "z": z,
            "count": (labels != IGNORE_INDEX).sum(),
        }
