"""CUDA kernel (kernels/softmax_gradient.cu): cross-entropy statistics and the logit
gradient in place, one block per row.

For each row of bf16 or fp32 logits and its target, pass one computes the
log-normalizer `lse` with an online max/sum in fp32; pass two overwrites the row
with the gradient of `ce + z_loss * lse^2`: `softmax * (1 + 2 z_loss lse) -
onehot(target)`, or zeros for an ignored target (-100). The row is read twice and
written once in its own dtype, instead of materializing fp32 copies of the
[rows, vocab] block. Registered as a custom op so torch.compile treats it as one node.
"""

import torch

from .cuda_extension import load


@torch.library.custom_op("hybrid::softmax_gradient_", mutates_args=("logits",))
def softmax_gradient_(
    logits: torch.Tensor, targets: torch.Tensor, z_loss: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite `logits` with its gradient; return fp32 (lse, target logit) per row."""
    lse, target_logit = load("softmax_gradient").softmax_gradient(
        logits, targets.contiguous(), z_loss
    )
    return lse, target_logit


@softmax_gradient_.register_fake
def _(logits, targets, z_loss):
    stats = logits.new_empty(logits.shape[0], dtype=torch.float32)
    return stats, torch.empty_like(stats)
