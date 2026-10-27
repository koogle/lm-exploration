"""Fused q/k RoPE and its adjoint, retaining the KDA [B,T,H,D] layout."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def rotate_pair(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    N: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    offset = (i // HALF) * (2 * HALF) + i % HALF
    position = (i // (H * HALF)) * HALF + i % HALF
    c = tl.load(C + position, valid, other=0).to(tl.float32)
    s = tl.load(S + position, valid, other=0).to(tl.float32)
    if BACKWARD:
        s = -s
    qa = tl.load(Q + offset, valid, other=0).to(tl.float32)
    qb = tl.load(Q + offset + HALF, valid, other=0).to(tl.float32)
    ka = tl.load(K + offset, valid, other=0).to(tl.float32)
    kb = tl.load(K + offset + HALF, valid, other=0).to(tl.float32)
    tl.store(OQ + offset, qa * c - qb * s, valid)
    tl.store(OQ + offset + HALF, qb * c + qa * s, valid)
    tl.store(OK + offset, ka * c - kb * s, valid)
    tl.store(OK + offset + HALF, kb * c + ka * s, valid)


def launch(q, k, cos, sin, backward):
    q, k = q.contiguous(), k.contiguous()
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    half = q.shape[-1] // 2
    n = q.numel() // 2
    rotate_pair[(tr.cdiv(n, 1024),)](
        q,
        k,
        cos,
        sin,
        oq,
        ok,
        n,
        q.shape[-2],
        half,
        backward,
        1024,
        enable_fp_fusion=False,  # Preserve separate FP32 product rounding in the reference.
    )
    return oq, ok


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.is_cuda and k.is_cuda and q.shape == k.shape and q.dtype == k.dtype
        assert q.ndim == 4 and q.dtype in (torch.float32, torch.bfloat16)
        assert q.shape[-1] % 2 == 0
        assert not cos.requires_grad and not sin.requires_grad
        assert cos.numel() == sin.numel() == q.shape[0] * q.shape[1] * q.shape[-1] // 2
        cos, sin = cos.contiguous(), sin.contiguous()
        ctx.save_for_backward(cos, sin)
        return launch(q, k, cos, sin, False)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        gq, gk = launch(dq, dk, cos, sin, True)
        return gq, gk, None, None


def rope_pair(q, k, cos, sin):
    return _RoPE.apply(q, k, cos, sin)
