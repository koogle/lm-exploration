"""CUDA kernel for the fixed hard-target cross-entropy; see kernels/stable_ce.cu.

The training image prebuilds the extension into PREBUILT, next to a copy of the
source it compiled; that build is used only when the copy matches this checkout's
source exactly. Otherwise it is JIT-built with torch.utils.cpp_extension, which needs
nvcc matching the installed torch. The kernel is registered as a custom op so
torch.compile treats it as one opaque node.
"""

import functools
import importlib.util
import os
from pathlib import Path

import torch
from training_data.subword import LOSS_IGNORE_INDEX

SOURCE = Path(__file__).with_name("kernels") / "stable_ce.cu"
PREBUILT = Path(os.environ.get("STABLE_CE_PREBUILT", "/opt/stable_ce"))


@functools.cache
def extension():
    compiled_source = PREBUILT / "stable_ce.cu"
    if compiled_source.exists() and compiled_source.read_bytes() == SOURCE.read_bytes():
        spec = importlib.util.spec_from_file_location("stable_ce", PREBUILT / "stable_ce.so")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load

    return load(name="stable_ce", sources=[str(SOURCE)], extra_cuda_cflags=["-O3"])


def _accumulator(logits):
    return torch.float64 if logits.dtype == torch.float64 else torch.float32


@torch.library.custom_op("lm_exploration::stable_ce_forward", mutates_args=())
def _forward(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    loss, lse = extension().forward(logits, targets, ignore_index)
    return loss, lse


@_forward.register_fake
def _(logits, targets, ignore_index):
    rows = logits.shape[0]
    dtype = _accumulator(logits)
    return logits.new_empty(rows, dtype=dtype), logits.new_empty(rows, dtype=dtype)


@torch.library.custom_op("lm_exploration::stable_ce_backward", mutates_args=())
def _backward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lse: torch.Tensor,
    grad_loss: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return extension().backward(logits, targets, lse, grad_loss, ignore_index)


@_backward.register_fake
def _(logits, targets, lse, grad_loss, ignore_index):
    return torch.empty_like(logits, memory_format=torch.contiguous_format)


def _setup_context(ctx, inputs, output):
    logits, targets, ignore_index = inputs
    ctx.save_for_backward(logits, targets, output[1])
    ctx.ignore_index = ignore_index


def _autograd_backward(ctx, grad_loss, _grad_lse):
    logits, targets, lse = ctx.saved_tensors
    return _backward(logits, targets, lse, grad_loss, ctx.ignore_index), None, None


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def cross_entropy(logits, targets, *, ignore_index=LOSS_IGNORE_INDEX, reduction="mean"):
    """Same contract as `trainer.stable_softmax.cross_entropy` for CUDA (N, V) logits."""
    loss, _ = _forward(logits.contiguous(), targets.contiguous(), ignore_index)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / (targets != ignore_index).sum()
    raise ValueError(f"Unknown reduction: {reduction}")
