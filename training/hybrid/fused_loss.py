"""Output layer, cross-entropy and z-loss in one pass, gradients computed in forward.

For each chunk of rows the forward computes logits h W^T, the log-normalizer
lse, the summed cross-entropy and squared lse, and the gradient
`dlogits = p * (1 + 2 z_loss lse) - onehot(target)` (zero for ignored targets). It
immediately reduces that to the hidden-state gradient `dlogits W` and accumulates
the weight gradient `dlogits^T h` in fp32, so the [rows, vocab] logits are never
stored or recomputed. The backward only scales the saved gradients by the
incoming gradient of the summed loss. Matmuls run in bf16 on CUDA (as under
autocast) and in the input precision elsewhere. On CUDA the per-row statistics
and the logit gradient come from our CUDA kernel (softmax_kernel.py) that
overwrites the bf16 logits in place; elsewhere PyTorch computes them in fp32.
"""

import torch

IGNORE_INDEX = -100


def _loss_grad(hidden, weight, labels, z_loss, chunk_size, grad_weight, scale=None):
    compute = torch.bfloat16 if hidden.is_cuda else hidden.dtype
    accumulate = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    rows = hidden.reshape(-1, hidden.size(-1))
    targets = labels.reshape(-1)
    w = weight.to(compute)
    coefficients = (
        torch.stack((scale.float(), torch.ones_like(scale).float()))
        if scale is not None and hidden.is_cuda
        else None
    )
    grad_hidden = torch.empty(rows.shape, dtype=compute, device=rows.device)
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
            from .output_gradient import accumulate_output_gradient_

            if scale is None:
                accumulate_output_gradient_(grad_weight, grad, h)
            else:
                from .output_gradient import accumulate_scaled_output_gradient_

                accumulate_scaled_output_gradient_(grad_weight, grad, h, coefficients)
        else:
            update = (grad.T @ h).to(accumulate)
            grad_weight += update if scale is None else update * scale
    return grad_hidden, ce, z


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, z_loss, chunk_size):
        accumulate = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        grad_weight = torch.zeros(weight.shape, dtype=accumulate, device=weight.device)
        grad_hidden, ce, z = _loss_grad(hidden, weight, labels, z_loss, chunk_size, grad_weight)
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


@torch.library.custom_op("hybrid::joint_loss_forward", mutates_args=())
def _joint_forward(
    hidden: torch.Tensor,
    future: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    mtp_labels: torch.Tensor,
    ntp_scale: torch.Tensor,
    mtp_scale: torch.Tensor,
    z_loss: float,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Opaque compiler boundary: retain the shared buffer without unrolling both heads into FX."""
    accumulate = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    grad_weight = torch.zeros_like(weight, dtype=accumulate)
    first, ce, z = _loss_grad(hidden, weight, labels, z_loss, chunk_size, grad_weight, ntp_scale)
    second, mtp_ce, mtp_z = _loss_grad(
        future, weight, mtp_labels, z_loss, chunk_size, grad_weight, mtp_scale
    )
    objective = ntp_scale * (ce + z_loss * z) + mtp_scale * (mtp_ce + z_loss * mtp_z)
    return objective, ce, z, mtp_ce, mtp_z, first, second, grad_weight


@_joint_forward.register_fake
def _(hidden, future, weight, labels, mtp_labels, ntp_scale, mtp_scale, z_loss, chunk_size):
    accumulate = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    compute = torch.bfloat16 if hidden.is_cuda else hidden.dtype
    stats = tuple(hidden.new_empty((), dtype=accumulate) for _ in range(5))
    first = hidden.new_empty((hidden.numel() // hidden.shape[-1], hidden.shape[-1]), dtype=compute)
    second = future.new_empty((future.numel() // future.shape[-1], future.shape[-1]), dtype=compute)
    return *stats, first, second, torch.empty_like(weight, dtype=accumulate)


class JointLinearCrossEntropy(torch.autograd.Function):
    """Two weighted objectives sharing one FP32 output-weight gradient buffer."""

    @staticmethod
    def forward(
        ctx, hidden, future, weight, labels, mtp_labels, ntp_scale, mtp_scale, z_loss, chunk_size
    ):
        objective, ce, z, mtp_ce, mtp_z, first, second, grad_weight = _joint_forward(
            hidden, future, weight, labels, mtp_labels, ntp_scale, mtp_scale, z_loss, chunk_size
        )
        ctx.save_for_backward(first, second, grad_weight, ntp_scale, mtp_scale)
        ctx.hidden_meta = ((hidden.shape, hidden.dtype), (future.shape, future.dtype))
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, z, mtp_ce, mtp_z)
        return objective, ce, z, mtp_ce, mtp_z

    @staticmethod
    def backward(ctx, upstream, _ce, _z, _mtp_ce, _mtp_z):
        first, second, weight, ntp_scale, mtp_scale = ctx.saved_tensors
        gradients = []
        for gradient, scale, (shape, dtype) in zip(
            (first, second), (ntp_scale, mtp_scale), ctx.hidden_meta, strict=True
        ):
            gradients.append((gradient.to(dtype) * (upstream * scale).to(dtype)).reshape(shape))
        return (
            *gradients,
            (weight * upstream).to(ctx.weight_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def joint_cross_entropy(hidden, future, weight, labels, mtp_labels, scales, z_loss, chunk_size):
    return JointLinearCrossEntropy.apply(
        hidden, future, weight, labels, mtp_labels, *scales, z_loss, chunk_size
    )
