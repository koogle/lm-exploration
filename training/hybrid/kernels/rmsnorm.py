"""FP32 residual-stream RMSNorm with fused row gradients and grouped weight reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0)
    w = tl.load(W + col, col < N, other=0)
    r = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(Y + row * N + col, x * r * w, col < N)
    tl.store(R + row, r)


@triton.jit
def _backward(
    X, W, DY, R, DX, DW, M: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr
):
    group = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    w = tl.load(W + col, col < N, other=0)
    dw = tl.full((BLOCK,), 0, tl.float32)
    for row in range(group, M, GROUPS):
        x = tl.load(X + row * N + col, col < N, other=0)
        dy = tl.load(DY + row * N + col, col < N, other=0)
        r = tl.load(R + row)
        normalized = x * r
        weighted = dy * w
        dx = (weighted - normalized * (tl.sum(normalized * weighted, 0) / N)) * r
        tl.store(DX + row * N + col, dx, col < N)
        dw += dy * normalized
    tl.store(DW + group * N + col, dw, col < N)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x, weight = x.contiguous(), weight.contiguous()
        width = x.shape[-1]
        rows = x.numel() // width
        y = torch.empty_like(x)
        r = x.new_empty(rows)
        _forward[(rows,)](x, weight, y, r, width, eps, triton.next_power_of_2(width), num_warps=4)
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, r = ctx.saved_tensors
        width = x.shape[-1]
        rows = x.numel() // width
        groups = min(rows, 512)
        dx = torch.empty_like(x)
        partial = x.new_empty((groups, width))
        _backward[(groups,)](
            x,
            weight,
            dy.contiguous(),
            r,
            dx,
            partial,
            rows,
            width,
            groups,
            triton.next_power_of_2(width),
            num_warps=4,
        )
        return dx, partial.sum(0), None


def rms_norm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)
