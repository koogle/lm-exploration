"""Training cross-entropy: the CUDA kernel on GPU, the PyTorch port elsewhere.

Both avoid Softmax Collapse (see ../references/softmax-fix-2026-10-24.md). Half and
bf16 logits are computed in fp32, as F.cross_entropy does under autocast.
"""

from training_data.subword import LOSS_IGNORE_INDEX

from trainer import stable_softmax


def cross_entropy(logits, targets, *, ignore_index=LOSS_IGNORE_INDEX, reduction="mean"):
    if logits.is_cuda:
        from trainer import stable_ce_cuda

        return stable_ce_cuda.cross_entropy(
            logits, targets, ignore_index=ignore_index, reduction=reduction
        )
    return stable_softmax.cross_entropy(
        logits, targets, ignore_index=ignore_index, reduction=reduction
    )
