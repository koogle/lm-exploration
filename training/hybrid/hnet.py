"""Byte-level H-Net with one stage of dynamic chunking around the hybrid trunk.

Follows "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (Hwang,
Wang and Gu, https://arxiv.org/abs/2507.07955) and its reference code
(github.com/goombalab/hnet, `hnet/modules/dc.py`, `hnet/models/hnet.py`):

- Bytes (ids 0-255, EOS 256) are embedded and run through an encoder of
  `outer_layers` hybrid blocks at width `outer_embd`.
- The routing module scores each position with p_t = (1 - cos(q(x_(t-1)), k(x_t))) / 2,
  q and k initialized to the identity; the first byte of every document has p = 1.
  Positions with p > 0.5 are chunk boundaries.
- The chunk layer keeps the boundary positions, pads them to a learned vector of
  width `main.n_embd` and runs the main network (the hybrid trunk: KDA and full
  attention, document-masked over chunks) on them.
- The dechunk layer smooths the main network's outputs with an EMA,
  z_t = p_t h_t + (1 - p_t) z_(t-1) over chunks, and repeats each chunk's value over
  its bytes. The result times STE(selected probability) plus a zero-initialized fp32
  projection of the encoder output forms the decoder input.
- A decoder of `outer_layers` hybrid blocks and an untied byte output layer predict
  the next byte. The ratio loss, weight `ratio_loss_weight`, pulls the fraction of
  boundaries towards 1 / `target_ratio`.

Every stage restarts at each EOS: attention, KDA state, convolutions, routing and
the EMA never cross a document boundary. The released 2025 code uses Mamba-2
outer layers; here the outer stage uses the family's KDA blocks, so the whole
model runs on the kernels the 720M hybrid already uses.
"""

import math
from dataclasses import dataclass, field, fields

import torch
from model_10m.model import RMSNorm
from torch import nn
from torch.nn import functional as F

from trainer.roles import declare

from .documents import document_ids
from .model import Block, DocumentMask, HybridConfig, chunked_cross_entropy, rope_tables

__all__ = [
    "BYTE_EOS",
    "BYTE_VOCAB",
    "ByteTokenizer",
    "HNetConfig",
    "HNetLM",
    "dechunk_ema",
    "flops_per_byte",
    "static_compilation",
]

BYTE_EOS = 256
BYTE_VOCAB = 257
IGNORE_INDEX = -100
# Selected chunks have p > 0.5, so each step of the EMA keeps less than half of the
# previous value; 2^-128 is below fp32 resolution, so two blocks of 128 chunks are exact.
EMA_BLOCK = 128


@dataclass(frozen=True)
class HNetConfig:
    """Outer (byte) stage settings plus the main network's HybridConfig fields."""

    main: dict = field(default_factory=dict)
    block_size: int = 16384
    outer_embd: int = 1024
    outer_layers: int = 4
    outer_kda_head: int = 8
    outer_mlp_hidden: int = 2816
    target_ratio: float = 4.5
    ratio_loss_weight: float = 0.03
    chunk_bucket: int = 512
    z_loss: float = 1e-4
    loss_chunk_size: int = 8192
    vocab_size: int = BYTE_VOCAB
    eos_id: int = BYTE_EOS
    mtp: bool = False

    def __post_init__(self):
        if self.vocab_size != BYTE_VOCAB or self.eos_id != BYTE_EOS:
            raise ValueError("H-Net reads UTF-8 bytes with EOS 256")
        if self.mtp:
            raise ValueError("H-Net has no MTP module")
        if self.target_ratio <= 1 or self.ratio_loss_weight < 0:
            raise ValueError("target_ratio must exceed 1 and ratio_loss_weight be non-negative")
        if min(self.outer_layers, self.chunk_bucket) <= 0:
            raise ValueError("outer_layers and chunk_bucket must be positive")
        if self.block_size % self.chunk_bucket:
            raise ValueError("block_size must be a multiple of chunk_bucket")
        # Validates the main network's own fields.
        self.main_config()
        self.outer_config()

    def main_config(self):
        known = {f.name for f in fields(HybridConfig)}
        settings = {k: tuple(v) if isinstance(v, list) else v for k, v in self.main.items()}
        if set(settings) - known:
            raise ValueError(f"Unknown main-network fields {sorted(set(settings) - known)}")
        return HybridConfig(
            **{
                "vocab_size": 1,
                "eos_id": 0,
                "mtp": False,
                **settings,
                "block_size": self.block_size,
            }
        )

    def outer_config(self):
        """Encoder/decoder blocks: KDA mixers at the byte width, sharing the main head size."""
        main = self.main_config()
        if self.outer_embd > main.n_embd:
            raise ValueError("outer_embd must not exceed the main network's width")
        return HybridConfig(
            vocab_size=1,
            eos_id=0,
            block_size=self.block_size,
            n_layer=self.outer_layers,
            n_embd=self.outer_embd,
            head_dim=main.head_dim,
            mlp_hidden=self.outer_mlp_hidden,
            layer_pattern=("kda",),
            n_kda_head=self.outer_kda_head,
            kda_conv_size=main.kda_conv_size,
            mtp=False,
            norm_eps=main.norm_eps,
            init_std=main.init_std,
        )


class ByteTokenizer:
    """UTF-8 bytes with the tokenizers-library interface the trainer's samples use."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self.Encoding(list(text.encode()))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


class STE(torch.autograd.Function):
    """Ones in the forward pass; the gradient passes through unchanged."""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class RoutingModule(nn.Module):
    """Boundary probability from the cosine similarity of adjacent encoder states."""

    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(self.q_proj.in_features))
            self.k_proj.weight.copy_(torch.eye(self.k_proj.in_features))

    def optimizer(self):
        # AdamW, as in the paper: with Muon the boundary rate swung between hundreds of
        # bytes per chunk and about 2 in 1-GPU tests (plans/HNET_720M.md).
        yield from declare("adamw_no_decay", self.q_proj, self.k_proj)

    def forward(self, x, positions):
        """[B, T] fp32 probability that each position starts a chunk."""
        query = F.normalize(self.q_proj(x[:, :-1]).float(), dim=-1)
        key = F.normalize(self.k_proj(x[:, 1:]).float(), dim=-1)
        probability = ((1 - (query * key).sum(-1)) / 2).clamp(0, 1)
        probability = F.pad(probability, (1, 0), value=1.0)
        return torch.where(positions == 0, 1.0, probability)


def chunk_order(boundaries, bucket):
    """Indices [B, M] that move each row's boundaries to its front, and their count [B].

    M is the largest count rounded up to a multiple of `bucket`, so compiled shapes
    repeat; positions past a row's count are padding.
    """
    length = boundaries.size(1)
    counts = boundaries.sum(1)
    width = min(length, -(-int(counts.max()) // bucket) * bucket)
    index = torch.arange(length, device=boundaries.device)
    order = torch.argsort(index + (~boundaries).long() * length, dim=1, stable=True)
    return order[:, :width], counts


def dechunk_ema(values, probability, documents):
    """z_t = p_t v_t + (1 - p_t) z_(t-1) along dim 1, restarting at each new document.

    values [B, M, D]; probability [B, M] in (0.5, 1); documents [B, M]. Each block of
    EMA_BLOCK chunks reads itself and the previous block; older chunks carry a
    weight below 2^-EMA_BLOCK.
    """
    batch, length, width = values.shape
    blocks = -(-length // EMA_BLOCK)
    pad = blocks * EMA_BLOCK - length
    # Padding is its own document, so it never mixes into real chunks.
    p = F.pad(
        probability.to(torch.promote_types(values.dtype, torch.float32)),
        (EMA_BLOCK, pad),
        value=0.75,
    )
    doc = F.pad(documents, (EMA_BLOCK, pad), value=-1)
    doc = torch.where(doc < 0, documents.max() + 1, doc)
    v = F.pad(values, (0, 0, EMA_BLOCK, pad))
    windows = 2 * EMA_BLOCK
    # Each window holds the previous block and the current one.
    p = p.unfold(1, windows, EMA_BLOCK)  # [B, blocks, 2C]
    doc = doc.unfold(1, windows, EMA_BLOCK)
    v = v.unfold(1, windows, EMA_BLOCK).transpose(-1, -2)  # [B, blocks, 2C, D]
    log_keep = torch.log1p(-p).cumsum(-1)
    current = slice(EMA_BLOCK, windows)
    decay = log_keep[..., current, None] - log_keep[..., None, :]  # [B, blocks, C, 2C]
    index = torch.arange(windows, device=values.device)
    causal = index[None, :] <= index[current, None]
    same = doc[..., current, None] == doc[..., None, :]
    weights = torch.where(causal & same, decay.clamp(max=0).exp() * p[..., None, :], 0.0)
    out = weights.to(v.dtype) @ v
    return out.reshape(batch, blocks * EMA_BLOCK, width)[:, :length]


class HNetLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        main, outer = config.main_config(), config.outer_config()
        self.main_settings, self.outer_settings = main, outer
        self.byte_embedding = nn.Embedding(config.vocab_size, config.outer_embd)
        self.encoder = nn.ModuleList(Block(outer, "kda") for _ in range(config.outer_layers))
        self.encoder_norm = RMSNorm(config.outer_embd, main.norm_eps)
        self.routing = RoutingModule(config.outer_embd)
        self.residual_proj = nn.Linear(config.outer_embd, config.outer_embd, bias=False)
        self.pad_dimension = nn.Parameter(torch.zeros(main.n_embd - config.outer_embd))
        self.main = nn.ModuleList(Block(main, kind) for kind in main.layer_kinds)
        self.main_norm = RMSNorm(main.n_embd, main.norm_eps)
        self.decoder = nn.ModuleList(Block(outer, "kda") for _ in range(config.outer_layers))
        self.decoder_norm = RMSNorm(config.outer_embd, main.norm_eps)
        self.lm_head = nn.Linear(config.outer_embd, config.vocab_size, bias=False)
        self.apply(self.initialize)
        self.routing.reset_parameters()
        nn.init.zeros_(self.residual_proj.weight)
        cos, sin = rope_tables(config.block_size, main.head_dim, main.rope_base, None)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def output_weight(self):
        return self.lm_head.weight

    def optimizer(self):
        # Blocks and the routing module declare their own parameters.
        yield from declare("muon", self.residual_proj)
        yield from declare("adamw_decay", self.byte_embedding, self.lm_head)
        yield from declare(
            "adamw_no_decay",
            self.encoder_norm,
            self.pad_dimension,
            self.main_norm,
            self.decoder_norm,
        )

    def initialize(self, module):
        """Normal(0, 0.02); residual output projections scaled by 1/sqrt(2 * all layers)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.main_settings.init_std
            if getattr(module, "residual_output", False):
                layers = self.main_settings.n_layer + 2 * self.config.outer_layers
                std /= math.sqrt(2 * layers)
            nn.init.normal_(module.weight, std=std)

    def chunks(self, x, documents, positions):
        """Routing, chunking and the main network; returns its output per byte and stats."""
        probability = self.routing(x, positions)
        boundaries = probability > 0.5
        order, counts = chunk_order(boundaries, self.config.chunk_bucket)
        width = order.size(1)
        valid = torch.arange(width, device=x.device)[None, :] < counts[:, None]
        # Padding chunks form one extra document after the row's last one.
        chunk_documents = torch.where(
            valid, documents.gather(1, order), documents.max(1, keepdim=True).values + 1
        )
        hidden = x.gather(1, order[..., None].expand(-1, -1, x.size(-1)))
        hidden = torch.cat(
            (hidden, self.pad_dimension.to(hidden.dtype).expand(*order.shape, -1)), -1
        )
        mask = DocumentMask(chunk_documents)
        rope = (self.rope_cos[mask.positions][:, None], self.rope_sin[mask.positions][:, None])
        for block in self.main:
            hidden = block(hidden, rope, mask)
        hidden = self.main_norm(hidden)[..., : x.size(-1)]
        selected = probability.gather(1, order).clamp(1e-4, 1 - 1e-4)
        selected = torch.where(valid, selected, 1 - 1e-4)
        smoothed = dechunk_ema(hidden, selected, chunk_documents)
        expand = (boundaries.cumsum(1) - 1).clamp_min(0)
        per_byte = smoothed.gather(1, expand[..., None].expand(-1, -1, x.size(-1)))
        chosen = torch.where(boundaries, probability, 1 - probability)
        return per_byte, chosen, probability, boundaries

    def ratio_loss(self, probability, boundaries):
        """N / (N - 1) * ((N - 1) F G + (1 - F)(1 - G)); 1 when F = G = 1 / N."""
        n = self.config.target_ratio
        selected = boundaries.float().mean()
        mean_probability = probability.mean()
        return (
            n
            / (n - 1)
            * ((n - 1) * selected * mean_probability + (1 - selected) * (1 - mean_probability))
        )

    def hidden_states(self, tokens):
        return self.forward_hidden(tokens)[0]

    def forward_hidden(self, tokens):
        length = tokens.size(1)
        if not 0 < length <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        documents = document_ids(tokens, self.config.eos_id)
        mask = DocumentMask(documents)
        x = self.byte_embedding(tokens)
        for block in self.encoder:
            x = block(x, None, mask)
        x = self.encoder_norm(x)
        # The residual path runs in fp32 (fp64 in tests) outside autocast.
        with torch.autocast(x.device.type, enabled=False):
            residual = self.residual_proj(x.to(self.residual_proj.weight.dtype))
        per_byte, chosen, probability, boundaries = self.chunks(x, documents, mask.positions)
        x = (per_byte.to(residual.dtype) * STE.apply(chosen)[..., None] + residual).to(x.dtype)
        for block in self.decoder:
            x = block(x, None, mask)
        return self.decoder_norm(x), probability, boundaries

    def forward(self, tokens, labels=None, mtp_labels=None):
        """Logits, or summed losses when `labels` are given.

        Returns {"loss", "ce", "z", "count", "ratio", "chunks", "positions"}: `loss`
        is ce + z_loss * z + ratio_loss_weight * ratio * count, so dividing it by
        the global valid-target count adds the weighted mean ratio loss to the
        mean next-byte loss.
        """
        if mtp_labels is not None:
            raise ValueError("H-Net has no MTP module")
        hidden, probability, boundaries = self.forward_hidden(tokens)
        if labels is None:
            return self.lm_head(hidden)
        ce, z = chunked_cross_entropy(
            hidden, self.lm_head.weight, labels, self.config.loss_chunk_size
        )
        count = (labels != IGNORE_INDEX).sum()
        ratio = self.ratio_loss(probability, boundaries)
        loss = ce + self.config.z_loss * z + self.config.ratio_loss_weight * ratio * count
        return {
            "loss": loss,
            "ce": ce,
            "z": z,
            "count": count,
            "ratio": ratio.detach(),
            "chunks": boundaries.sum(),
            "positions": torch.tensor(boundaries.numel(), device=tokens.device),
        }


def static_compilation(config):
    """Compile one static graph per chunk bucket instead of a dynamic-shape graph.

    The main network's length is the batch's largest chunk count rounded up to
    `chunk_bucket`, so torch.compile sees a few lengths; dynamic shapes through
    FlexAttention and fla fail to compile (torch 2.14.1).
    """
    torch._dynamo.config.automatic_dynamic_shapes = False
    buckets = config.block_size // config.chunk_bucket
    limits = torch._dynamo.config
    limits.cache_size_limit = max(limits.cache_size_limit, buckets)
    limits.accumulated_cache_size_limit = max(limits.accumulated_cache_size_limit, 4 * buckets)


def flops_per_byte(lm, bytes_per_chunk, attended_chunks):
    """Training FLOPs per byte: outer matmuls every byte, the main network every chunk."""
    outer = sum(
        p.numel()
        for name, p in lm.named_parameters()
        if p.ndim == 2 and not name.startswith("main.") and name != "byte_embedding.weight"
    )
    main = sum(p.numel() for p in lm.main.parameters() if p.ndim == 2)
    settings = lm.main_settings
    full_layers = settings.layer_kinds.count("full")
    attention = 12 * full_layers * settings.n_head * settings.head_dim * attended_chunks
    return 6 * outer + (6 * main + attention) / bytes_per_chunk
