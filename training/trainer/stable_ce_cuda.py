"""CUDA kernel for the fixed hard-target cross-entropy; see kernels/stable_ce.cu.

The extension is JIT-built on first use with torch.utils.cpp_extension, which needs
nvcc (a CUDA toolkit image) matching the installed torch.
"""

import functools
from pathlib import Path

import torch
from training_data.subword import LOSS_IGNORE_INDEX


@functools.cache
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("kernels") / "stable_ce.cu"
    return load(name="stable_ce", sources=[str(source)], extra_cuda_cflags=["-O3"])


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        loss, lse = _extension().forward(logits, targets, ignore_index)
        ctx.save_for_backward(logits, targets, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, lse = ctx.saved_tensors
        grad = _extension().backward(logits, targets, lse, grad_loss, ctx.ignore_index)
        return grad, None, None


def cross_entropy(logits, targets, *, ignore_index=LOSS_IGNORE_INDEX, reduction="mean"):
    """Same contract as `trainer.stable_softmax.cross_entropy` for CUDA (N, V) logits."""
    loss = _CrossEntropy.apply(logits.contiguous(), targets.contiguous(), ignore_index)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / (targets != ignore_index).sum()
    raise ValueError(f"Unknown reduction: {reduction}")
