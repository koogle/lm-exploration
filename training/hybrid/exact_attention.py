"""Fixed-order definition of document-causal attention, the CUDA kernel's reference.

Every value is a float32 computed by a fixed sequence of correctly rounded
multiplies, adds and divides, so any implementation that follows the same order
(this PyTorch emulation, kernels/document_attention.cu, a later Metal port) gives
bit-identical results. There is no fused multiply-add and no library exp or log:
`exp_fixed` and `log_fixed` use only rounded arithmetic and exact bit operations.

With s_ij = seqsum_d(q_id * k_jd) * scale over the allowed keys of row i
(start_i <= j <= i, the row's document):

    m_i = max_j s_ij                 e_ij = exp_fixed(s_ij - m_i)
    l_i = seqsum_j e_ij              o_id = seqsum_j(e_ij * v_jd) / l_i
    lse_i = m_i + log_fixed(l_i)

`seqsum` adds terms one at a time in index order, starting from +0. Disallowed
entries contribute exact zeros, which leave a sum unchanged, so kernels may skip
them. The backward, with D_i = seqsum_d(dout_id * out_id) using the rounded
output, recomputes p_ij = exp_fixed(s_ij - m_i) / l_i and
dp_ij = seqsum_d(dout_id * v_jd), then

    ds_ij = p_ij * ((dp_ij - D_i) + dlse_i)
    dq_id = seqsum_j(ds_ij * k_jd) * scale
    dk_jd = seqsum_(g, i)(ds_ij * q_id) * scale     dv_jd = seqsum_(g, i)(p_ij * dout_id)

where (g, i) runs over the query heads sharing key/value head k (GQA) in order,
then over query rows in order. Outputs and gradients round to the input dtype.
"""

import torch

__all__ = ["document_starts", "exact_attention", "exact_attention_backward", "exp_fixed"]


def _f32(hex_value):
    return torch.tensor(float.fromhex(hex_value), dtype=torch.float32)


LOG2E = _f32("0x1.715476p+0")
LN2 = _f32("0x1.62e430p-1")
SQRT2 = _f32("0x1.6a09e6p+0")
# 2**f = exp(f ln 2) on |f| <= 1/2: Taylor coefficients (ln 2)^k / k!, k = 7 .. 1.
EXP_COEFFICIENTS = [
    _f32(h)
    for h in (
        "0x1.ffcbfcp-17",
        "0x1.430912p-13",
        "0x1.5d87fep-10",
        "0x1.3b2ab6p-7",
        "0x1.c6b08ep-5",
        "0x1.ebfbe0p-3",
        "0x1.62e430p-1",
        "0x1p+0",
    )
]
# log(m) = 2 atanh(s), s = (m - 1) / (m + 1): coefficients 1/9, 1/7, 1/5, 1/3, 1.
LOG_COEFFICIENTS = [
    _f32(h) for h in ("0x1.c71c72p-4", "0x1.24924ap-3", "0x1.99999ap-3", "0x1.555556p-2", "0x1p+0")
]
EXP_UNDERFLOW = -125.0


def exp_fixed(x):
    """exp(x) for float32 x <= 0, about 1e-7 relative error; 0 below 2**-125."""
    y = x * LOG2E
    n = torch.round(torch.clamp(y, min=EXP_UNDERFLOW))  # Round half to even, like rintf.
    f = y - n
    p = torch.zeros_like(f) + EXP_COEFFICIENTS[0]
    for coefficient in EXP_COEFFICIENTS[1:]:
        p = p * f + coefficient
    power = ((n.to(torch.int32) + 127) << 23).view(torch.float32)
    return torch.where(y >= EXP_UNDERFLOW, p * power, 0.0)


def log_fixed(x):
    """log(x) for finite normal float32 x > 0."""
    bits = x.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = ((bits & 0x7FFFFF) | 0x3F800000).view(torch.float32)
    large = mantissa > SQRT2
    mantissa = torch.where(large, mantissa * 0.5, mantissa)
    exponent = exponent + large.to(torch.int32)
    s = (mantissa - 1.0) / (mantissa + 1.0)
    s2 = s * s
    t = torch.zeros_like(s) + LOG_COEFFICIENTS[0]
    for coefficient in LOG_COEFFICIENTS[1:]:
        t = t * s2 + coefficient
    return exponent.to(torch.float32) * LN2 + (s * t) * 2.0


def document_starts(documents):
    """int32 [B, T] index of the first token of each token's document."""
    index = torch.arange(documents.size(1), device=documents.device).expand_as(documents)
    start = torch.ones_like(documents, dtype=torch.bool)
    start[:, 1:] = documents[:, 1:] != documents[:, :-1]
    return torch.where(start, index, 0).cummax(1).values.to(torch.int32).contiguous()


def _seq_dot(a, b):
    """[..., M, D] x [..., N, D] -> [..., M, N], summing over D in order."""
    total = torch.zeros(*a.shape[:-1], b.size(-2), dtype=torch.float32, device=a.device)
    for d in range(a.size(-1)):
        total = total + a[..., :, None, d] * b[..., None, :, d]
    return total


def _seq_rows(weights, values):
    """[..., M, N] x [..., N, D] -> [..., M, D], summing over N in order."""
    total = torch.zeros(*weights.shape[:-1], values.size(-1), dtype=torch.float32)
    total = total.to(weights.device)
    for j in range(weights.size(-1)):
        total = total + weights[..., :, j, None] * values[..., None, j, :]
    return total


def _allowed(starts, length):
    index = torch.arange(length, device=starts.device)
    keys, rows = index[None, None, None, :], index[None, None, :, None]
    return (keys <= rows) & (keys >= starts[:, None, :, None])


def _scores(q, k, starts, scale):
    groups = q.size(1) // k.size(1)
    s = _seq_dot(q.float(), k.float().repeat_interleave(groups, 1))
    s = s * torch.tensor(scale, dtype=torch.float32)
    return s, _allowed(starts, q.size(2))


def exact_attention(q, k, v, starts, scale):
    """q [B, H, T, D], k/v [B, KVH, T, D], starts [B, T] -> out, lse, m, l."""
    groups = q.size(1) // k.size(1)
    s, allowed = _scores(q, k, starts, scale)
    m = torch.where(allowed, s, -torch.inf).amax(-1)
    e = torch.where(allowed, exp_fixed(torch.where(allowed, s - m[..., None], 0.0)), 0.0)
    total = torch.zeros_like(m)
    for j in range(e.size(-1)):
        total = total + e[..., j]
    weighted = _seq_rows(e, v.float().repeat_interleave(groups, 1))
    out = (weighted / total[..., None]).to(q.dtype)
    return out, m + log_fixed(total), m, total


def exact_attention_backward(q, k, v, starts, scale, out, dout, dlse, m, total):
    """Gradients (dq, dk, dv) for exact_attention, in the documented order."""
    batch, heads, length, width = q.shape
    kv_heads = k.size(1)
    groups = heads // kv_heads
    s, allowed = _scores(q, k, starts, scale)
    dout = dout.float()
    delta = torch.zeros_like(m)
    for d in range(width):
        delta = delta + dout[..., d] * out.float()[..., d]
    shifted = torch.where(allowed, s - m[..., None], 0.0)
    p = torch.where(allowed, exp_fixed(shifted) / total[..., None], 0.0)
    dp = _seq_dot(dout, v.float().repeat_interleave(groups, 1))
    ds = torch.where(allowed, p * ((dp - delta[..., None]) + dlse.float()[..., None]), 0.0)
    scale = torch.tensor(scale, dtype=torch.float32)
    dq = _seq_rows(ds, k.float().repeat_interleave(groups, 1)) * scale

    grouped = (batch, kv_heads, groups, length)
    ds, p = ds.view(*grouped, length), p.view(*grouped, length)
    qg, doutg = q.float().view(*grouped, width), dout.view(*grouped, width)
    dk = torch.zeros(batch, kv_heads, length, width, device=q.device)
    dv = torch.zeros_like(dk)
    for g in range(groups):
        for i in range(length):
            dk = dk + ds[:, :, g, i, :, None] * qg[:, :, g, i, None, :]
            dv = dv + p[:, :, g, i, :, None] * doutg[:, :, g, i, None, :]
    return dq.to(q.dtype), (dk * scale).to(k.dtype), dv.to(v.dtype)
