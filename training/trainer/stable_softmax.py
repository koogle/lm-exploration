"""Softmax and cross-entropy without Softmax Collapse or Softmax Cancellation.

PyTorch port of Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and
Cross-Entropy" (arXiv:2610.31642, https://jonbarron.info/softmaxfix/), equations 5-9.

Standard implementations add e^-m to 1.0, so once the logit margin m passes the
precision cliff (about 16.6 in fp32) the target gradient rounds to zero while the
non-target gradients stay positive. Here the largest (or target) logit is replaced
by an exact 0 and left out of the log-sum-exp; autodiff then gives the target the
negated sum of the other gradients. bf16/fp16 logits are upcast to fp32, as
F.cross_entropy does under autocast.
"""

import torch
from torch.nn import functional as F
from training_data.subword import LOSS_IGNORE_INDEX


def _upcast(logits):
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _margin(z, index):
    """Shifted logits with an exact 0 at `index`, and the margin of that slot (eq. 2)."""
    is_pivot = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, index.unsqueeze(-1), True)
    shifted = torch.where(is_pivot, 0.0, z - z.gather(-1, index.unsqueeze(-1)))
    margin = -torch.logsumexp(shifted.masked_fill(is_pivot, float("-inf")), -1, keepdim=True)
    return shifted, margin


def log_softmax(logits, dim=-1):
    z = _upcast(logits).transpose(dim, -1)
    shifted, margin = _margin(z, z.argmax(-1))
    return (shifted - F.softplus(-margin)).transpose(dim, -1)


def softmax(logits, dim=-1):
    return log_softmax(logits, dim).exp()


def cross_entropy(logits, targets, *, ignore_index=LOSS_IGNORE_INDEX, reduction="mean"):
    """Hard-target cross-entropy (eq. 6) for (N, V) logits and (N,) targets.

    Matches F.cross_entropy's ignore_index and reduction semantics.
    """
    z = _upcast(logits)
    valid = targets != ignore_index
    _, margin = _margin(z, torch.where(valid, targets, 0))
    loss = torch.where(valid, F.softplus(-margin.squeeze(-1)), 0.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / valid.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def soft_target_cross_entropy(student_logits, teacher_logits, dim=-1):
    """Per-row -sum p log q with p = softmax(teacher), q = softmax(student) (eq. 9).

    The backward pass uses q - p computed in log space from the logit difference,
    so the gradient is exactly zero when student and teacher logits are equal.
    The teacher is treated as frozen: no gradient reaches `teacher_logits`.
    """
    z_s = _upcast(student_logits).transpose(dim, -1)
    z_t = _upcast(teacher_logits).detach().transpose(dim, -1)
    log_q = log_softmax(z_s)
    with torch.no_grad():
        q, p = log_q.exp(), softmax(z_t)
        delta = z_s - z_t
        d = delta - (p * delta).sum(-1, keepdim=True)
        s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))  # log(q / p)
        # Large |s| can pair p == 0 with an overflowing expm1; fall back to q - p.
        u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)
        g = u - q * u.sum(-1, keepdim=True)
    loss = -(p * log_q.detach()).sum(-1)
    return loss + (g * (z_s - z_s.detach())).sum(-1)
