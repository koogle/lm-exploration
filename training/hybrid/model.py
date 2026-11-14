"""Hybrid-family decoder: Kimi Delta Attention and full-attention layers.

Pre-norm blocks reuse RMSNorm (fp32 statistics) and SwiGLU from model_10m, the
SwiGLU without biases at a configurable width; tied embeddings and no biases.
Full-attention layers use GQA with head dimension 128, QK-norm and RoPE with a
configurable base. Every forward pass masks attention by document: documents
end in EOS (`HybridConfig.eos_id`) and a token sees only earlier tokens of its
own document. On CUDA attention is compiled FlexAttention with that document-causal
block mask; on CPU an explicit masked softmax. The loss is a chunked cross-entropy
with a z-loss.
"""

import math
from dataclasses import dataclass

import torch
from model_10m.model import RMSNorm, SwiGLU
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from trainer.roles import declare

from .documents import cu_seqlens, document_ids, document_positions
from .fused_loss import fused_cross_entropy
from .kda import KimiDeltaAttention

__all__ = [
    "Attention",
    "Block",
    "DocumentMask",
    "HybridConfig",
    "HybridLM",
    "RMSNorm",
    "ShortConvolution",
    "SwiGLU",
    "chunked_cross_entropy",
]
IGNORE_INDEX = -100
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


class DocumentMask:
    """Document-aware causal mask: position q sees k <= q within its own document."""

    share_kda_metadata = False

    def __init__(self, documents):
        self.documents = documents
        self.positions = document_positions(documents)
        self._block = None
        self._kda_offsets = None

    @property
    def kda_offsets(self):
        """Int64 packed-document offsets shared by recurrent layers in this forward."""
        if not self.share_kda_metadata:
            return cu_seqlens(self.documents).long()
        if self._kda_offsets is None:
            self._kda_offsets = cu_seqlens(self.documents).long()
        return self._kda_offsets

    @property
    def block(self):
        """FlexAttention block mask, built once per batch on first use."""
        if self._block is None:
            documents = self.documents

            def allowed(batch, head, q, k):
                return (q >= k) & (documents[batch, q] == documents[batch, k])

            batch, length = documents.shape
            self._block = create_block_mask(allowed, batch, None, length, length, device="cuda")
        return self._block

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
    """Attention output [B, H, T, D] and fp32 (or fp64) log-sum-exp [B, H, T].

    CUDA runs compiled FlexAttention (tensor cores) with the document-causal block
    mask; tests/check_hybrid_cuda.py compares it with `reference_attention` in fp64.
    """
    if not query.is_cuda:
        return reference_attention(query, key, value, mask.dense())
    output, aux = compiled_flex_attention(
        query,
        key,
        value,
        block_mask=mask.block,
        enable_gqa=key.size(1) != query.size(1),
        return_aux=AuxRequest(lse=True),
    )
    return output, aux.lse


POSITIVE_FIELDS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_embd",
    "n_head",
    "n_kv_head",
    "head_dim",
    "mlp_hidden",
    "n_kda_head",
    "kda_conv_size",
    "rope_base",
    "norm_eps",
    "init_std",
    "loss_chunk_size",
)


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
    layer_pattern: tuple[str, ...] = ("kda", "kda", "kda", "full")
    n_kda_head: int = 12
    kda_conv_size: int = 4
    mtp: bool = True
    mtp_layer_kind: str = "full"
    mtp_weight_start: float = 0.3
    mtp_weight_end: float = 0.1
    mtp_loss_fraction: float = 1.0
    gate_bias_init: float = 3.0
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    z_loss: float = 1e-4
    loss_chunk_size: int = 8192
    # Looping (Elastic Latent Recurrence, arXiv:2610.12582): blocks
    # [loop_prelude, n_layer - loop_coda) form one shared recurrent group, applied
    # n_loops times by default. n_loops == 1 is the plain stack.
    loop_prelude: int = 0
    loop_coda: int = 0
    n_loops: int = 1
    # Converting a trained plain model: the loop state starts at zero instead of the
    # prelude output and the step size at 1, so one loop computes the plain stack.
    loop_retrofit: bool = False
    # Recompute each KDA block in backward instead of storing its activations (training
    # only), for contexts whose activations do not fit, such as 40,960 tokens on an H100.
    # Full-attention blocks keep theirs: recomputed under torch.compile, FlexAttention
    # falls back to dense attention, which needs [T, T] scores.
    activation_checkpointing: bool = False
    # Study options. `short_conv_size` adds causal depthwise convolutions of that width
    # (identity at initialization, never reading an earlier document) on full-attention
    # keys and values, on each full-attention layer's output and on every MLP output.
    # `embedding_norm` applies an RMSNorm to the embedding lookup.
    short_conv_size: int = 0
    embedding_norm: bool = False

    def __post_init__(self):
        for name in POSITIVE_FIELDS:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.z_loss, self.mtp_weight_start, self.mtp_weight_end) < 0:
            raise ValueError("z_loss and MTP weights must be non-negative")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError("eos_id must be a token id")
        if not self.layer_pattern or set(self.layer_pattern) - {"kda", "full"}:
            raise ValueError("layer_pattern must list 'kda' and 'full' layers")
        if not 0 < self.mtp_loss_fraction <= 1:
            raise ValueError("mtp_loss_fraction must be in (0, 1]")
        if self.mtp_layer_kind not in ("kda", "full"):
            raise ValueError("mtp_layer_kind must be 'kda' or 'full'")
        if self.n_layer % len(self.layer_pattern):
            raise ValueError("n_layer must be a multiple of the layer pattern length")
        if min(self.loop_prelude, self.loop_coda) < 0 or self.n_loops < 1:
            raise ValueError("loop_prelude and loop_coda must be >= 0 and n_loops >= 1")
        if self.n_loops > 1 and self.loop_prelude + self.loop_coda >= self.n_layer:
            raise ValueError("a looped model needs at least one recurrent block")
        if self.n_loops == 1 and (self.loop_prelude + self.loop_coda or self.loop_retrofit):
            raise ValueError("loop_prelude, loop_coda and loop_retrofit need n_loops > 1")

    @property
    def layer_kinds(self):
        return tuple(self.layer_pattern[i % len(self.layer_pattern)] for i in range(self.n_layer))

    @property
    def looped(self):
        return self.n_loops > 1

    @property
    def recurrent_blocks(self):
        """Indices of the shared recurrent blocks (all blocks when not looped)."""
        return range(self.loop_prelude, self.n_layer - self.loop_coda)

    def block_uses(self, loops=None):
        """How many times each block runs per token at `loops` (default n_loops)."""
        loops = self.n_loops if loops is None else loops
        recurrent = self.recurrent_blocks
        return tuple(loops if i in recurrent else 1 for i in range(self.n_layer))


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


class ShortConvolution(nn.Module):
    """Causal depthwise convolution over time, per channel, within each document.

    `y[t] = sum_j w[:, j] * x[t - j]` over the taps whose source position lies in the
    same document. The weight starts as the identity (`w[:, 0] = 1`), so adding the
    module to a trained model leaves its function unchanged.
    """

    def __init__(self, channels, size):
        super().__init__()
        weight = torch.zeros(channels, size)
        weight[:, 0] = 1.0
        self.weight = nn.Parameter(weight)

    def forward(self, x, positions):
        """x [B, T, C]; positions [B, T], each token's index within its document."""
        weight = self.weight.to(x.dtype)
        out = x * weight[:, 0]
        for tap in range(1, weight.size(1)):
            shifted = F.pad(x[:, :-tap], (0, 0, tap, 0))
            out = out + shifted * weight[:, tap] * (positions >= tap)[..., None].to(x.dtype)
        return out


class Attention(nn.Module):
    """Document-causal GQA with QK-norm, RoPE, an attention sink and an output gate.

    The sink is one learned logit per head for an extra key with a zero value:
    softmax over [scores, sink] scales the output by sigmoid(lse - sink). The
    elementwise gate sigmoid(W_g x + b) is computed from the layer input; `b` is
    the family's only bias and starts at `gate_bias_init` (3.0: about 95% open).
    """

    # Set during evaluation to record gate openness and sink weight per layer.
    record_statistics = False

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
        self.sink = nn.Parameter(torch.zeros(config.n_head))
        self.gate_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.gate_bias = nn.Parameter(
            torch.full((config.n_head * config.head_dim,), config.gate_bias_init)
        )
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        self.o_proj.residual_output = True
        self.convolutions = None
        if config.short_conv_size:
            kv = config.n_kv_head * config.head_dim
            self.convolutions = nn.ModuleDict(
                {
                    "key": ShortConvolution(kv, config.short_conv_size),
                    "value": ShortConvolution(kv, config.short_conv_size),
                    "output": ShortConvolution(config.n_embd, config.short_conv_size),
                }
            )

    def optimizer(self):
        yield from declare(
            "muon", self.q_proj, self.k_proj, self.v_proj, self.gate_proj, self.o_proj
        )
        yield from declare("adamw_no_decay", self.q_norm, self.k_norm, self.sink, self.gate_bias)
        if self.convolutions is not None:
            yield from declare("adamw_no_decay", self.convolutions)

    def project(self, x, rope, positions=None):
        batch, length, _ = x.shape
        query = self.q_proj(x).view(batch, length, self.n_head, self.head_dim)
        key, value = self.k_proj(x), self.v_proj(x)
        if self.convolutions is not None:
            if positions is None:
                raise ValueError("Short convolutions need document positions")
            key = self.convolutions["key"](key, positions)
            value = self.convolutions["value"](value, positions)
        key = key.view(batch, length, self.n_kv_head, self.head_dim)
        value = value.view(batch, length, self.n_kv_head, self.head_dim)
        query, key = self.q_norm(query), self.k_norm(key)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        cos, sin = rope
        return apply_rope(query, cos, sin), apply_rope(key, cos, sin), value

    def forward(self, x, rope, mask):
        query, key, value = self.project(x, rope, mask.positions)
        attended, lse = attend(query, key, value, mask)
        # lse is fp32 from FlexAttention, and fp32/fp64 from the reference.
        kept = torch.sigmoid(lse - self.sink.to(lse.dtype)[:, None])
        attended = (attended * kept[..., None]).to(attended.dtype)
        batch, length, _ = x.shape
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        gate = torch.sigmoid(self.gate_proj(x) + self.gate_bias)
        if Attention.record_statistics:
            self.statistics = {
                "gate_open": gate.float().mean().item(),
                "sink_weight": (1 - kept).float().mean().item(),
            }
        out = self.o_proj(attended * gate.to(attended.dtype))
        if self.convolutions is not None:
            out = self.convolutions["output"](out, mask.positions)
        return out


class Block(nn.Module):
    """Pre-norm block whose token mixer is KDA or full attention."""

    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        self.mixer_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mixer = KimiDeltaAttention(config) if kind == "kda" else Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config.n_embd, config.mlp_hidden, bias=False)
        self.mlp.down.residual_output = True
        self.mlp_convolution = None
        if config.short_conv_size:
            self.mlp_convolution = ShortConvolution(config.n_embd, config.short_conv_size)

    def optimizer(self):
        # The mixer declares its own parameters.
        yield from declare("muon", self.mlp)
        yield from declare("adamw_no_decay", self.mixer_norm, self.mlp_norm)
        if self.mlp_convolution is not None:
            yield from declare("adamw_no_decay", self.mlp_convolution)

    def forward(self, x, rope, mask):
        x = x + self.mixer(self.mixer_norm(x), rope, mask)
        out = self.mlp(self.mlp_norm(x))
        if self.mlp_convolution is not None:
            out = self.mlp_convolution(out, mask.positions)
        return x + out


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


class MultiTokenPrediction(nn.Module):
    """DeepSeek-V3/GLM-style MTP module predicting token t+2 at position t.

    Combines RMSNorm(h_t), the trunk's final hidden state before its own norm,
    with RMSNorm(embedding(x_(t+1))), projects back to the model width and runs
    one hybrid block. The shared (tied) output layer reads its normalized output.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.embedding_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.fuse = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.block = Block(config, config.mtp_layer_kind)
        self.output_norm = RMSNorm(config.n_embd, config.norm_eps)

    def optimizer(self):
        # The block declares its own parameters.
        yield from declare("muon", self.fuse)
        yield from declare(
            "adamw_no_decay", self.hidden_norm, self.embedding_norm, self.output_norm
        )

    def forward(self, hidden, next_embeddings, rope, mask):
        fused = torch.cat((self.hidden_norm(hidden), self.embedding_norm(next_embeddings)), -1)
        return self.output_norm(self.block(self.fuse(fused), rope, mask))


def mtp_weight(config, progress):
    """MTP loss weight decaying linearly from start to end over training progress [0, 1]."""
    progress = min(max(progress, 0.0), 1.0)
    return config.mtp_weight_start + (config.mtp_weight_end - config.mtp_weight_start) * progress


class LoopUpdate(nn.Module):
    """ELR's adaptive-delta state update, shared by every loop:

        beta = sigmoid(W entry + b) * softplus(gamma)
        state <- state + beta * (proposal - state)

    `entry` is the recurrent blocks' input (state plus the re-injected prelude
    output) and `proposal` their output. W, b and gamma start at zero, so every
    token and channel starts with step size softplus(0) / 2 ~ 0.35. In the paper's
    ablations this is the only update rule that matches the unlooped baseline at one
    loop; replacing the state with the proposal, or adding it, does worse.

    With `initial_step = 1` (a converted plain model) gamma starts at
    softplus^-1(2), so the first loop's state is exactly the recurrent blocks' output.
    """

    def __init__(self, width, initial_step=None):
        super().__init__()
        self.gate = nn.Linear(width, width)
        self.rate = nn.Parameter(torch.zeros(width))
        self.initial_rate = 0.0 if initial_step is None else math.log(math.expm1(2 * initial_step))

    def reset_parameters(self):
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.constant_(self.rate, self.initial_rate)

    def optimizer(self):
        yield from declare("muon", self.gate.weight)
        yield from declare("adamw_no_decay", self.gate.bias, self.rate)

    def forward(self, state, proposal, entry):
        step = torch.sigmoid(self.gate(entry)) * F.softplus(self.rate)
        return state + step * (proposal - state)


class HybridLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config, kind) for kind in config.layer_kinds)
        self.final_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.embedding_norm = (
            RMSNorm(config.n_embd, config.norm_eps) if config.embedding_norm else None
        )
        self.loop_update = None
        if config.looped:
            self.loop_update = LoopUpdate(config.n_embd, 1.0 if config.loop_retrofit else None)
        self.mtp = MultiTokenPrediction(config) if config.mtp else None
        self.apply(self.initialize)
        cos, sin = rope_tables(config.block_size, config.head_dim, config.rope_base, None)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _apply(self, fn, recurse=True):
        # RoPE tables keep at least fp32 precision when the model is cast to bf16/fp16.
        super()._apply(fn, recurse)
        if self.rope_cos.dtype in (torch.bfloat16, torch.float16):
            config, device = self.config, self.rope_cos.device
            tables = rope_tables(config.block_size, config.head_dim, config.rope_base, device)
            self.rope_cos, self.rope_sin = tables
        return self

    @property
    def output_weight(self):
        return self.token_embedding.weight

    def optimizer(self):
        # Blocks and the MTP module declare their own parameters. The tied
        # embedding/output table has per-token rows, so it stays on AdamW.
        yield from declare("adamw_decay", self.token_embedding)
        yield from declare("adamw_no_decay", self.final_norm)
        if self.embedding_norm is not None:
            yield from declare("adamw_no_decay", self.embedding_norm)

    def initialize(self, module):
        """Normal(0, 0.02); residual output projections scaled by 1/sqrt(2 * n_layer).

        The loop update's gate starts at zero (see `LoopUpdate`)."""
        if isinstance(module, LoopUpdate):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.init_std
            if getattr(module, "residual_output", False):
                std /= math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, std=std)

    def trunk(self, tokens, loops=None):
        """Residual stream after the last block, with the shared mask and RoPE tables.

        Each EOS ends a document: attention, convolutions, KDA state and RoPE
        positions restart after it. A looped model runs its prelude once, the
        recurrent blocks `loops` times (default `n_loops`) and its coda once. The
        state starts as the prelude output `h` (zero with `loop_retrofit`); each loop
        re-injects `h`, runs the recurrent blocks on `state + h` and takes a gated step
        towards their output (`LoopUpdate`).
        """
        length = tokens.size(1)
        if not 0 < length <= self.config.block_size:
            raise ValueError("sequence length must be within the model's block_size")
        documents = document_ids(tokens, self.config.eos_id)
        mask = DocumentMask(documents)
        rope = (self.rope_cos[mask.positions][:, None], self.rope_sin[mask.positions][:, None])
        x = self.token_embedding(tokens)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        config = self.config
        if not config.looped:
            if loops not in (None, 1):
                raise ValueError("loops applies to looped models only")
            for block in self.blocks:
                x = self.run_block(block, x, rope, mask)
            return x, rope, mask
        loops = config.n_loops if loops is None else loops
        if loops < 1:
            raise ValueError("loops must be at least 1")
        recurrent = config.recurrent_blocks
        for block in self.blocks[: recurrent.start]:
            x = self.run_block(block, x, rope, mask)
        injected = x
        state = torch.zeros_like(x) if config.loop_retrofit else x
        for _ in range(loops):
            x = entry = state + injected
            for block in self.blocks[recurrent.start : recurrent.stop]:
                x = self.run_block(block, x, rope, mask)
            state = self.loop_update(state, x, entry)
        x = state
        for block in self.blocks[recurrent.stop :]:
            x = self.run_block(block, x, rope, mask)
        return x, rope, mask

    def run_block(self, block, x, rope, mask):
        """One block; a KDA block recomputes in backward under `activation_checkpointing`."""
        if (
            self.config.activation_checkpointing
            and self.training
            and block.kind == "kda"
            and torch.is_grad_enabled()
        ):
            return checkpoint(block, x, rope, mask, use_reentrant=False)
        return block(x, rope, mask)

    def hidden_states(self, tokens, loops=None):
        return self.final_norm(self.trunk(tokens, loops)[0])

    def loss_sums(self, hidden, labels, prefix=""):
        if hidden.is_cuda and torch.is_grad_enabled():
            loss, ce, z = fused_cross_entropy(
                hidden, self.output_weight, labels, self.config.z_loss, self.config.loss_chunk_size
            )
        else:
            ce, z = chunked_cross_entropy(
                hidden, self.output_weight, labels, self.config.loss_chunk_size
            )
            loss = ce + self.config.z_loss * z
        return {
            f"{prefix}loss": loss,
            f"{prefix}ce": ce,
            f"{prefix}z": z,
            f"{prefix}count": (labels != IGNORE_INDEX).sum(),
        }

    def forward(self, tokens, labels=None, mtp_labels=None, loops=None, loss_scales=None):
        """Logits, or summed losses when `labels` are given.

        `loops` sets a looped model's recurrent loop count for this call (default
        `n_loops`); the same weights serve every loop count.

        Returns {"loss", "ce", "z", "count"}: `loss = ce + z_loss * z` summed over
        valid targets; divide by the global valid-target count to get the mean.
        With `mtp_labels` (targets for t+2, see `documents.mtp_labels`), also
        returns the same sums for the MTP module under an `mtp_` prefix. Without
        them the MTP module does not run, as in inference. In training mode with
        `mtp_loss_fraction < 1`, the MTP module runs over every position but only a
        uniformly random `mtp_loss_fraction` of them (fixed count) go through the
        output layer and loss; divide `mtp_loss` by the expected valid count
        (all valid MTP targets times the fraction).
        """
        x, rope, mask = self.trunk(tokens, loops)
        hidden = self.final_norm(x)
        if labels is None:
            return F.linear(hidden, self.output_weight)
        losses = self.loss_sums(hidden, labels) if loss_scales is None else {}
        if mtp_labels is not None:
            if self.mtp is None:
                raise ValueError("this model was built without an MTP module")
            # x_(t+1) for every position; the last one is the final label. Masked
            # labels never feed a valid MTP target, so any in-vocabulary id works.
            next_tokens = torch.cat((tokens[:, 1:], labels[:, -1:].clamp_min(0)), 1)
            future = self.mtp(x, self.token_embedding(next_tokens), rope, mask)
            if self.training and self.config.mtp_loss_fraction < 1:
                future, mtp_labels = future.flatten(0, 1), mtp_labels.flatten()
                keep = max(1, round(future.size(0) * self.config.mtp_loss_fraction))
                # Indices of the `keep` largest uniform keys: a random subset, without
                # randperm (which torch.compile cannot trace).
                rows = torch.rand(future.size(0), device=future.device).topk(keep).indices
                future, mtp_labels = future[rows], mtp_labels[rows]
            if loss_scales is None:
                losses |= self.loss_sums(future, mtp_labels, "mtp_")
            else:
                from .fused_loss import joint_cross_entropy

                objective, ce, z, mtp_ce, mtp_z = joint_cross_entropy(
                    hidden,
                    future,
                    self.output_weight,
                    labels,
                    mtp_labels,
                    loss_scales,
                    self.config.z_loss,
                    self.config.loss_chunk_size,
                )
                losses = {
                    "objective": objective,
                    "ce": ce,
                    "z": z,
                    "mtp_ce": mtp_ce,
                    "mtp_z": mtp_z,
                }
        elif loss_scales is not None:
            raise ValueError("Joint objective requires MTP targets")
        return losses
