"""CUDA kernel (kernels/softmax_gradient.cu): cross-entropy statistics and the logit
gradient in place, one block per row.

For each row of bf16 or fp32 logits and its target, pass one computes the
log-normalizer `lse` with an online max/sum in fp32; pass two overwrites the row
with the gradient of `ce + z_loss * lse^2`: `softmax * (1 + 2 z_loss lse) -
onehot(target)`, or zeros for an ignored target (-100). The row is read twice and
written once in its own dtype, instead of materializing fp32 copies of the
[rows, vocab] block. Registered as a custom op so torch.compile treats it as one node.

The image prebuilds the extension into PREBUILT next to a copy of its source; that
build is used only when the copy matches this checkout's source exactly. Otherwise
it is JIT-built, which needs nvcc matching the installed torch.
"""

import functools
import importlib.util
import os
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("kernels") / "softmax_gradient.cu"
PREBUILT = Path(os.environ.get("HYBRID_SOFTMAX_PREBUILT", "/opt/hybrid_softmax"))


@functools.cache
def extension():
    compiled_source = PREBUILT / SOURCE.name
    if compiled_source.exists() and compiled_source.read_bytes() == SOURCE.read_bytes():
        path = PREBUILT / "hybrid_softmax_gradient.so"
        spec = importlib.util.spec_from_file_location("hybrid_softmax_gradient", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load

    return load(name="hybrid_softmax_gradient", sources=[str(SOURCE)], extra_cuda_cflags=["-O3"])


@torch.library.custom_op("hybrid::softmax_gradient_", mutates_args=("logits",))
def softmax_gradient_(
    logits: torch.Tensor, targets: torch.Tensor, z_loss: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite `logits` with its gradient; return fp32 (lse, target logit) per row."""
    lse, target_logit = extension().softmax_gradient(logits, targets.contiguous(), z_loss)
    return lse, target_logit


@softmax_gradient_.register_fake
def _(logits, targets, z_loss):
    stats = logits.new_empty(logits.shape[0], dtype=torch.float32)
    return stats, torch.empty_like(stats)
