"""Segmented depthwise causal convolution + SiLU, with analytic Triton gradients."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _forward(
    X,
    W,
    P,
    Z,
    Y,
    N: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = i % C
    t = i // C % T
    pos = tl.load(P + i // C, i < N, other=0)
    z = tl.full((B,), 0, tl.float32)
    for lag in tl.static_range(K):
        x = tl.load(X + i - lag * C, (i < N) & (t >= lag) & (pos >= lag), other=0).to(tl.float32)
        w = tl.load(W + c * K + K - 1 - lag)
        z = z + x * w
    tl.store(Z + i, z, i < N)
    tl.store(Y + i, z * tl.sigmoid(z), i < N)


@tr.jit
def _dz(Z, DY, DZ, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    z = tl.load(Z + i, i < N, other=0)
    s = tl.sigmoid(z)
    dy = tl.load(DY + i, i < N, other=0)
    tl.store(DZ + i, dy * s * (1 + z * (1 - s)), i < N)


@tr.jit
def _dx(
    DZ,
    W,
    P,
    DX,
    N: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c, t = i % C, i // C % T
    acc = tl.full((B,), 0, tl.float32)
    for lag in tl.static_range(K):
        valid = (i < N) & (t + lag < T)
        pos = tl.load(P + i // C + lag, valid, other=0)
        d = tl.load(DZ + i + lag * C, valid & (pos >= lag), other=0)
        w = tl.load(W + c * K + K - 1 - lag)
        # Match branch-wise BF16 rounding in PyTorch's input gradients.
        term = (d * w).to(DX.dtype.element_ty).to(tl.float32)
        acc = (acc + term).to(DX.dtype.element_ty).to(tl.float32)
    tl.store(DX + i, acc, i < N)


@tr.jit
def _dw(
    X,
    DZ,
    P,
    PART,
    R: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    NB: tl.constexpr,
    B: tl.constexpr,
):
    ck = tl.program_id(0)
    c, lag = ck // K, ck % K
    block = tl.program_id(1)
    r = block * B + tl.arange(0, B)
    pos = tl.load(P + r, r < R, other=0)
    x = tl.load(X + (r - lag) * C + c, (r < R) & (r % T >= lag) & (pos >= lag), other=0).to(
        tl.float32
    )
    dz = tl.load(DZ + r * C + c, r < R, other=0)
    tl.store(PART + (c * K + K - 1 - lag) * NB + block, tl.sum(x * dz, 0))


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, positions):
        x, weight, positions = x.contiguous(), weight.contiguous(), positions.contiguous()
        _b, t, c = x.shape
        z = torch.empty_like(x, dtype=torch.float32)
        y = torch.empty_like(z)
        _forward[(tr.cdiv(x.numel(), 256),)](
            x,
            weight,
            positions,
            z,
            y,
            x.numel(),
            t,
            c,
            weight.size(-1),
            256,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, positions, z)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, p, z = ctx.saved_tensors
        b, t, c = x.shape
        k, n = w.size(-1), x.numel()
        dz, dx = torch.empty_like(z), torch.empty_like(x)
        _dz[(tr.cdiv(n, 256),)](z, dy.contiguous(), dz, n, 256, enable_fp_fusion=False)
        _dx[(tr.cdiv(n, 256),)](dz, w, p, dx, n, t, c, k, 256, enable_fp_fusion=False)
        nb = tr.cdiv(b * t, 256)
        partial = torch.empty((c, k, nb), device=x.device, dtype=torch.float32)
        _dw[(c * k, nb)](x, dz, p, partial, b * t, t, c, k, nb, 256, enable_fp_fusion=False)
        return dx, partial.sum(-1).view_as(w), None


def local_conv(x, weight, positions):
    return _Conv.apply(x, weight, positions)
