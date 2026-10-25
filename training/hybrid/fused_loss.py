"""Output layer, cross-entropy and z-loss in one pass, gradients computed in forward.

For each chunk of rows the forward computes logits h W^T, the log-normalizer
lse, the summed cross-entropy and squared lse, and the gradient
`dlogits = p * (1 + 2 z_loss lse) - onehot(target)` (zero for ignored targets). It
immediately reduces that to the hidden-state gradient `dlogits W` and accumulates
the weight gradient `dlogits^T h` in fp32, so the [rows, vocab] logits are never
stored or recomputed. The backward only scales the saved gradients by the
incoming gradient of the summed loss. Matmuls run in bf16 on CUDA (as under
autocast) and in the input precision elsewhere. On CUDA the per-row statistics
and the logit gradient come from one Triton kernel (softmax_kernel.py) that
overwrites the bf16 logits in place; elsewhere PyTorch computes them in fp32.
"""

import torch

IGNORE_INDEX = -100


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, z_loss, chunk_size):
        compute = torch.bfloat16 if hidden.is_cuda else hidden.dtype
        accumulate = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        rows = hidden.reshape(-1, hidden.size(-1))
        targets = labels.reshape(-1)
        w = weight.to(compute)
        grad_hidden = torch.empty(rows.shape, dtype=compute, device=rows.device)
        grad_weight = torch.zeros(weight.shape, dtype=accumulate, device=weight.device)
        ce = torch.zeros((), dtype=accumulate, device=rows.device)
        z = torch.zeros((), dtype=accumulate, device=rows.device)
        for start in range(0, rows.size(0), chunk_size):
            h = rows[start : start + chunk_size].to(compute)
            y = targets[start : start + chunk_size]
            valid = y != IGNORE_INDEX
            index = y.clamp_min(0)[:, None]
            if h.is_cuda:
                from .softmax_kernel import softmax_gradient_

                grad = h @ w.T
                lse, target_logit = softmax_gradient_(grad, y, z_loss)
            else:
                logits = (h @ w.T).to(accumulate)
                lse = logits.logsumexp(-1)
                target_logit = logits.gather(-1, index).squeeze(-1)
                grad = (logits - lse[:, None]).exp_().mul_((1 + 2 * z_loss * lse)[:, None])
                grad.scatter_add_(-1, index, torch.full_like(lse[:, None], -1.0))
                grad = grad.mul_(valid[:, None]).to(compute)
            ce = ce + torch.where(valid, lse - target_logit, 0.0).sum()
            z = z + torch.where(valid, lse.square(), 0.0).sum()
            grad_hidden[start : start + chunk_size] = grad @ w
            if compute == torch.bfloat16 and grad.is_cuda:
                grad_weight += torch.mm(grad.T, h, out_dtype=torch.float32)
            else:
                grad_weight += (grad.T @ h).to(accumulate)
        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.hidden_meta = (hidden.shape, hidden.dtype)
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, z)
        return ce + z_loss * z, ce, z

    @staticmethod
    def backward(ctx, grad_loss, _grad_ce, _grad_z):
        grad_hidden, grad_weight = ctx.saved_tensors
        shape, dtype = ctx.hidden_meta
        grad_hidden = (grad_hidden.to(dtype) * grad_loss.to(dtype)).reshape(shape)
        grad_weight = (grad_weight * grad_loss.to(grad_weight.dtype)).to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None


def fused_cross_entropy(hidden, weight, labels, z_loss, chunk_size):
    """(ce + z_loss * z, ce, z) summed over valid targets."""
    return FusedLinearCrossEntropy.apply(hidden, weight, labels, z_loss, chunk_size)
