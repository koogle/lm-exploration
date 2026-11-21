"""Dynamic tensorwise FP8 cast without a full-size FP32 intermediate."""

import torch
import triton
import triton.language as tl


@triton.jit
def _amax(X, P, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    tl.store(P + tl.program_id(0), tl.max(tl.abs(x), 0))


@triton.jit
def _scale(P, S, N: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(P + i, i < N, other=0), 0)
    tl.store(S, tl.maximum(maximum, 1e-12) / LIMIT)


@triton.jit
def _cast(X, S, Y, N: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    scaled = tl.minimum(tl.maximum(x / tl.load(S), -LIMIT), LIMIT)
    tl.store(Y + i, scaled, i < N)


def quantize(x, dtype):
    x = x.contiguous()
    n = x.numel()
    partials = torch.empty(triton.cdiv(n, 16384), device=x.device, dtype=torch.float32)
    scale = torch.empty((), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x, dtype=dtype)
    limit = torch.finfo(dtype).max
    _amax[(partials.numel(),)](x, partials, n, 16384)
    _scale[(1,)](partials, scale, partials.numel(), limit, triton.next_power_of_2(partials.numel()))
    _cast[(triton.cdiv(n, 4096),)](x, scale, output, n, limit, 4096)
    return output, scale
