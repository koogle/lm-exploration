"""Kimi Delta Attention (KDA) layers for the hybrid family.

Follows the flash-linear-attention (fla) `KimiDeltaAttention` layer
(https://arxiv.org/abs/2510.26692) without biases: causal short convolutions
(kernel 4, SiLU) on q/k/v, L2-normalized q/k, sigmoid beta, a per-channel decay
gate `-exp(A_log) * softplus(f_proj(x) + dt_bias)`, and an output RMSNorm with a
sigmoid gate before the projection. There is no positional encoding.

The recurrence keeps an fp32 state that starts at zero in every document. On CUDA
it runs fla's chunked kernel over the flattened batch with `cu_seqlens`; on CPU an
explicit per-token recurrence is the reference. torch.compile traces through the
fla call.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .documents import cu_seqlens
from .roles import declare

__all__ = [
    "KimiDeltaAttention",
    "kda_kernel",
    "kda_reference",
    "short_convolution",
]


def short_convolution(x, weight, positions):
    """Causal depthwise convolution with SiLU that never reads across documents.

    x [B, T, C], weight [C, 1, K] (nn.Conv1d layout), positions [B, T] in-document.
    """
    width = weight.size(-1)
    out = x * weight[:, 0, -1]
    for shift in range(1, width):
        previous = F.pad(x, (0, 0, shift, 0))[:, : x.size(1)]
        previous = previous.masked_fill((positions < shift)[..., None], 0)
        out = out + previous * weight[:, 0, -1 - shift]
    return F.silu(out)


def kda_gate(g, A_log, dt_bias):
    """Log-space per-channel decay, [B, T, H, K] in fp32."""
    heads = A_log.numel()
    g = g.float() + dt_bias.float().view(heads, -1)
    return -A_log.float().exp()[:, None] * F.softplus(g)


def kda_reference(q, k, v, g, beta, A_log, dt_bias, positions):
    """Naive KDA recurrence; same raw inputs as `kda_kernel`, fp32 (or fp64) state.

    q, k [B, T, H, K]; v [B, T, H, V]; g raw gate [B, T, H, K]; beta raw logits
    [B, T, H]. The state resets wherever `positions` is 0.
    """
    dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(dtype), dim=-1) * q.size(-1) ** -0.5
    k = F.normalize(k.to(dtype), dim=-1)
    v, beta = v.to(dtype), torch.sigmoid(beta.to(dtype))
    decay = kda_gate(g, A_log, dt_bias).to(dtype).exp()
    batch, length, heads, key_width = k.shape
    state = q.new_zeros(batch, heads, key_width, v.size(-1))
    outputs = []
    for t in range(length):
        keep = (positions[:, t] > 0).to(dtype)[:, None, None, None]
        state = state * keep * decay[:, t, ..., None]
        k_t = k[:, t]
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, state)
        state = state + torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k_t, error)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outputs, 1)


def kda_kernel(q, k, v, g, beta, A_log, dt_bias, documents):
    """fla's chunked KDA over the flattened batch, one sequence per document."""
    from fla.ops.kda import chunk_kda

    batch, length = documents.shape
    offsets = cu_seqlens(documents).long()
    out, _ = chunk_kda(
        *(t.reshape(1, batch * length, *t.shape[2:]) for t in (q, k, v, g, beta)),
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        cu_seqlens=offsets,
    )
    return out.view(batch, length, *out.shape[2:])


class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads, width = config.n_kda_head, config.head_dim
        inner = heads * width
        self.heads, self.head_dim = heads, width
        self.q_proj = nn.Linear(config.n_embd, inner, bias=False)
        self.k_proj = nn.Linear(config.n_embd, inner, bias=False)
        self.v_proj = nn.Linear(config.n_embd, inner, bias=False)
        # Depthwise kernels in nn.Conv1d layout and default initialization.
        bound = 1 / math.sqrt(config.kda_conv_size)
        self.q_conv, self.k_conv, self.v_conv = (
            nn.Parameter(torch.empty(inner, 1, config.kda_conv_size).uniform_(-bound, bound))
            for _ in range(3)
        )
        self.f_proj = nn.Sequential(
            nn.Linear(config.n_embd, width, bias=False), nn.Linear(width, inner, bias=False)
        )
        self.b_proj = nn.Linear(config.n_embd, heads, bias=False)
        # Per-head decay rate in [1, 16] and per-channel time step in [0.001, 0.1].
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, width, bias=False), nn.Linear(width, inner, bias=False)
        )
        self.o_norm_weight = nn.Parameter(torch.ones(width))
        self.norm_eps = config.norm_eps
        self.o_proj = nn.Linear(inner, config.n_embd, bias=False)

    def optimizer_roles(self):
        yield from declare(
            "muon", self.q_proj, self.k_proj, self.v_proj, self.f_proj, self.g_proj, self.o_proj
        )
        # One row per head: a write strength per head, not a hidden matrix.
        yield from declare("adamw_decay", self.b_proj)
        yield from declare(
            "adamw_no_decay",
            self.q_conv,
            self.k_conv,
            self.v_conv,
            self.A_log,
            self.dt_bias,
            self.o_norm_weight,
        )
        self.o_proj.residual_output = True

    def forward(self, x, rope, mask):
        batch, length, _ = x.shape
        heads = (batch, length, self.heads, self.head_dim)
        positions = mask.positions
        q = short_convolution(self.q_proj(x), self.q_conv, positions).view(heads)
        k = short_convolution(self.k_proj(x), self.k_conv, positions).view(heads)
        v = short_convolution(self.v_proj(x), self.v_conv, positions).view(heads)
        g = self.f_proj(x).view(heads)
        beta = self.b_proj(x)
        if x.is_cuda:
            out = kda_kernel(q, k, v, g, beta, self.A_log, self.dt_bias, mask.documents)
        else:
            out = kda_reference(q, k, v, g, beta, self.A_log, self.dt_bias, positions)
        values = out.float() if out.dtype in (torch.float16, torch.bfloat16) else out
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.norm_eps)
        values = values * self.o_norm_weight.to(values.dtype)
        gate = torch.sigmoid(self.g_proj(x).view(heads).to(values.dtype))
        return self.o_proj((values * gate).to(x.dtype).reshape(batch, length, -1))
