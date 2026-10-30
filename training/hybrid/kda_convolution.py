"""KDA's three short convolutions as one CUDA kernel (kernels/kda_convolution.cu).

`kda_convolutions(xq, xk, xv, wq, wk, wv, positions)` returns
`short_convolution(x, w, positions)` for q, k and v (causal depthwise width-4
convolution, SiLU, never reading an earlier document) from one launch that reads
each input once. The backward recomputes the pre-activation instead of storing it
and returns input and fp32 weight gradients. Outputs keep the inputs' dtype
(bf16 under autocast); arithmetic is fp32. Registered as custom ops so
torch.compile treats each direction as one node.
"""

import torch

from .cuda_extension import load


@torch.library.custom_op("hybrid::kda_convolutions", mutates_args=())
def kda_convolutions(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [x.contiguous() for x in (xq, xk, xv)]
    ws = [w.float().contiguous() for w in (wq, wk, wv)]
    q, k, v = load("kda_convolution").forward(xs, ws, positions.contiguous())
    return q, k, v


@kda_convolutions.register_fake
def _(xq, xk, xv, wq, wk, wv, positions):
    return torch.empty_like(xq), torch.empty_like(xk), torch.empty_like(xv)


@torch.library.custom_op("hybrid::kda_convolutions_backward", mutates_args=())
def kda_convolutions_backward(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    positions: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [x.contiguous() for x in (xq, xk, xv)]
    ws = [w.float().contiguous() for w in (wq, wk, wv)]
    grads = [g.to(xs[0].dtype).contiguous() for g in (dq, dk, dv)]
    return tuple(load("kda_convolution").backward(xs, ws, positions.contiguous(), grads))


@kda_convolutions_backward.register_fake
def _(xq, xk, xv, wq, wk, wv, positions, dq, dk, dv):
    weights = [torch.empty_like(w, dtype=torch.float32) for w in (wq, wk, wv)]
    return (torch.empty_like(xq), torch.empty_like(xk), torch.empty_like(xv), *weights)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dq, dk, dv):
    xq, xk, xv, wq, wk, wv, positions = ctx.saved_tensors
    dxq, dxk, dxv, dwq, dwk, dwv = kda_convolutions_backward(
        xq, xk, xv, wq, wk, wv, positions, dq, dk, dv
    )
    weights = [d.to(w.dtype) for d, w in zip((dwq, dwk, dwv), (wq, wk, wv), strict=True)]
    return dxq, dxk, dxv, *weights, None


kda_convolutions.register_autograd(_backward, setup_context=_setup)
