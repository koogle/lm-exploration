"""Fused Triton kernel for the fixed hard-target cross-entropy (eqs. 6-7).

Each program owns one row. The forward pass streams the vocabulary once with an
online log-sum-exp of z_j - z_t over j != t and stores only that value per row.
With lse = LSE_{j != t}(z_j - z_t), the loss is softplus(lse) and the gradients are

    g_j = exp(z_j - z_t - softplus(lse))   for j != t,
    g_t = -exp(lse - softplus(lse)) = -sum_{j != t} g_j,

so nothing adds e^-m to 1.0 and no N x V intermediate is materialized beyond the
gradient. Logits are read in their own dtype and computed in fp32; the gradient
is written in the logits' dtype.
"""

import torch
import triton
import triton.language as tl
from training_data.subword import LOSS_IGNORE_INDEX


@triton.jit
def _softplus(x):
    # log1p without libdevice (unsupported by the interpreter): a series below
    # 1e-2 (relative error < 2e-9) never adds the small term to 1.0.
    y = tl.exp(-tl.abs(x))
    series = y * (1.0 - y * (0.5 - y * (1.0 / 3.0 - y * 0.25)))
    return tl.maximum(x, 0.0) + tl.where(y < 1e-2, series, tl.log(1.0 + y))


@triton.jit
def _forward(logits, stride, targets, lse_out, loss_out, vocab, ignore_index, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(targets + row)
    valid = target != ignore_index
    target = tl.where(valid, target, 0)
    base = logits + row * stride
    z_t = tl.load(base + target).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        keep = (cols < vocab) & (cols != target)
        x = tl.load(base + cols, mask=keep, other=-float("inf")).to(tl.float32) - z_t
        new_max = tl.maximum(running_max, tl.max(x, 0))
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(tl.exp(x - new_max), 0)
        running_max = new_max
    lse = running_max + tl.log(running_sum)
    tl.store(lse_out + row, lse)
    tl.store(loss_out + row, tl.where(valid, _softplus(lse), 0.0))


@triton.jit
def _backward(
    logits, stride, targets, lse_in, grad_loss, grad, grad_stride, vocab, ignore_index,
    BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(targets + row)
    valid = target != ignore_index
    target = tl.where(valid, target, 0)
    out = grad + row * grad_stride
    base = logits + row * stride
    z_t = tl.load(base + target).to(tl.float32)
    lse = tl.load(lse_in + row)
    loss = _softplus(lse)
    scale = tl.where(valid, tl.load(grad_loss + row).to(tl.float32), 0.0)
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        z = tl.load(base + cols, mask=cols < vocab, other=0.0).to(tl.float32)
        g = tl.where(cols == target, -tl.exp(lse - loss), tl.exp(z - z_t - loss))
        tl.store(out + cols, g * scale, mask=cols < vocab)


def _block(vocab):
    return min(4096, triton.next_power_of_2(vocab))


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        rows, vocab = logits.shape
        if vocab < 2 or logits.stride(-1) != 1:
            raise ValueError("Expected contiguous rows over a vocabulary of at least 2")
        lse = torch.empty(rows, device=logits.device, dtype=torch.float32)
        loss = torch.empty_like(lse)
        _forward[(rows,)](
            logits, logits.stride(0), targets, lse, loss, vocab, ignore_index, BLOCK=_block(vocab)
        )
        ctx.save_for_backward(logits, targets, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, lse = ctx.saved_tensors
        rows, vocab = logits.shape
        grad = torch.empty_like(logits)
        _backward[(rows,)](
            logits, logits.stride(0), targets, lse, grad_loss.contiguous(), grad, grad.stride(0),
            vocab, ctx.ignore_index, BLOCK=_block(vocab),
        )  # fmt: skip
        return grad, None, None


def cross_entropy(logits, targets, *, ignore_index=LOSS_IGNORE_INDEX, reduction="mean"):
    """Same contract as `trainer.stable_softmax.cross_entropy` for (N, V) logits."""
    targets = targets.contiguous()
    loss = _CrossEntropy.apply(logits, targets, ignore_index)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / (targets != ignore_index).sum()
    raise ValueError(f"Unknown reduction: {reduction}")
