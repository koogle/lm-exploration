"""Our document-causal GQA attention kernel (kernels/document_attention.cu).

Bit-exact with the fixed-order definition in exact_attention.py and deterministic:
no atomics, every float operation correctly rounded in a fixed order. Inputs are
bf16 or fp32, head dimension 64 or 128. Registered as custom ops so torch.compile
treats forward and backward as opaque nodes.

The image prebuilds the extension into PREBUILT next to a copy of its source; that
build is used only when the copy matches this checkout's source exactly. Otherwise
it is JIT-built, which needs nvcc matching the installed torch.
"""

import functools
import importlib.util
import math
import os
import struct
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("kernels") / "document_attention.cu"
PREBUILT = Path(os.environ.get("HYBRID_ATTENTION_PREBUILT", "/opt/hybrid_attention"))
CUDA_FLAGS = ["-O3", "-fmad=false"]


@functools.cache
def extension():
    compiled_source = PREBUILT / SOURCE.name
    if compiled_source.exists() and compiled_source.read_bytes() == SOURCE.read_bytes():
        path = PREBUILT / "hybrid_document_attention.so"
        spec = importlib.util.spec_from_file_location("hybrid_document_attention", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load

    return load(
        name="hybrid_document_attention", sources=[str(SOURCE)], extra_cuda_cflags=CUDA_FLAGS
    )


@torch.library.custom_op("hybrid::document_attention", mutates_args=())
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, starts: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out, lse, row_max, row_sum = extension().forward(q, k, v, starts, scale)
    return out, lse, row_max, row_sum


@_forward.register_fake
def _(q, k, v, starts, scale):
    stats = q.new_empty(q.shape[:3], dtype=torch.float32)
    return torch.empty_like(q), stats, torch.empty_like(stats), torch.empty_like(stats)


@torch.library.custom_op("hybrid::document_attention_backward", mutates_args=())
def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    starts: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    dlse: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = extension().backward(q, k, v, starts, out, dout, dlse, row_max, row_sum, scale)
    return dq, dk, dv


@_backward.register_fake
def _(q, k, v, starts, out, dout, dlse, row_max, row_sum, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, starts, scale = inputs
    out, _, row_max, row_sum = output
    ctx.save_for_backward(q, k, v, starts, out, row_max, row_sum)
    ctx.scale = scale
    ctx.mark_non_differentiable(row_max, row_sum)


def _autograd_backward(ctx, dout, dlse, _dmax, _dsum):
    q, k, v, starts, out, row_max, row_sum = ctx.saved_tensors
    dout = torch.zeros_like(out) if dout is None else dout.contiguous()
    dlse = torch.zeros_like(row_max) if dlse is None else dlse.float().contiguous()
    dq, dk, dv = _backward(q, k, v, starts, out, dout, dlse, row_max, row_sum, ctx.scale)
    return dq, dk, dv, None, None


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def softmax_scale(width):
    """1/sqrt(width) rounded to float32, the value the kernel and reference use."""
    return struct.unpack("f", struct.pack("f", 1 / math.sqrt(width)))[0]


def document_attention(q, k, v, starts, scale):
    """q [B, H, T, D], k/v [B, KVH, T, D], int32 document starts [B, T] -> (out, lse).

    `scale` must be a float32 value, such as softmax_scale(D).
    """
    out, lse, _, _ = _forward(q.contiguous(), k.contiguous(), v.contiguous(), starts, scale)
    return out, lse
