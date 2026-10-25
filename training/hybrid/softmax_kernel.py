"""Triton kernel: cross-entropy statistics and logit gradient in place, two passes per row.

For each row of `logits` (bf16 or fp32 [rows, vocab]) and its target, pass one
computes the log-normalizer `lse` with an online max/sum in fp32; pass two
overwrites the logits with the gradient of `ce + z_loss * lse^2`:
`softmax * (1 + 2 z_loss lse) - onehot(target)`, or zeros for an ignored target.
The row is read twice and written once in its own dtype, instead of
materializing several fp32 copies of the [rows, vocab] block.
"""

import triton
import triton.language as tl

IGNORE_INDEX = -100
BLOCK = 4096


@triton.jit
def _softmax_gradient(
    logits, targets, lse_out, target_out, z_loss, vocab, stride, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * stride
    target = tl.load(targets + row)
    maximum = tl.full((), float("-inf"), tl.float32)
    total = tl.zeros((), tl.float32)
    for start in range(0, vocab, BLOCK):
        columns = start + tl.arange(0, BLOCK)
        values = tl.load(base + columns, mask=columns < vocab, other=float("-inf")).to(tl.float32)
        new_maximum = tl.maximum(maximum, tl.max(values, 0))
        total = total * tl.exp(maximum - new_maximum) + tl.sum(tl.exp(values - new_maximum), 0)
        maximum = new_maximum
    lse = maximum + tl.log(total)
    valid = target != -100
    safe_target = tl.where(valid, target, 0)
    tl.store(lse_out + row, lse)
    tl.store(target_out + row, tl.load(base + safe_target).to(tl.float32))
    scale = tl.where(valid, 1 + 2 * z_loss * lse, 0.0)
    for start in range(0, vocab, BLOCK):
        columns = start + tl.arange(0, BLOCK)
        mask = columns < vocab
        values = tl.load(base + columns, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.exp(values - lse) * scale - tl.where(
            (columns == safe_target) & valid, 1.0, 0.0
        )
        tl.store(base + columns, gradient.to(logits.dtype.element_ty), mask=mask)


def softmax_gradient_(logits, targets, z_loss):
    """Overwrite `logits` with its gradient; return fp32 (lse, target logit) per row."""
    import torch

    rows, vocab = logits.shape
    if logits.stride(1) != 1:
        raise ValueError("logits must be contiguous along the vocabulary")
    lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
    target_logit = torch.empty_like(lse)
    _softmax_gradient[(rows,)](
        logits, targets, lse, target_logit, float(z_loss), vocab, logits.stride(0), BLOCK=BLOCK
    )
    return lse, target_logit
