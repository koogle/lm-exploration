"""Experimental mixed-precision vocabulary GEMMs, sharing the production loss loop."""

import torch

from .kernels.fp8_cast import quantize


class Projection:
    def __init__(self, weight, mode):
        self.weight, self.scale = quantize(weight, torch.float8_e4m3fn)
        self.mode = mode
        self.bf16_weight = weight
        # cuBLAS FP8 RHS must be column-major. Reuse both layouts across chunks.
        self.input_weight = self.weight.T.contiguous().T if mode in ("logits_dx", "dx") else None

    def logits(self, hidden):
        if self.mode == "dx":
            return hidden @ self.bf16_weight.T
        x, scale = quantize(hidden, torch.float8_e4m3fn)
        return torch._scaled_mm(
            x, self.weight.T, scale, self.scale, out_dtype=torch.bfloat16, use_fast_accum=True
        )

    def hidden_gradient(self, gradient, weight):
        if self.mode == "logits":
            return gradient @ weight
        x, scale = quantize(gradient, torch.float8_e5m2)
        return torch._scaled_mm(
            x, self.input_weight, scale, self.scale, out_dtype=torch.bfloat16, use_fast_accum=True
        )


@torch.library.custom_op("hybrid::fp8_loss_grad", mutates_args=("grad_weight",))
def loss_grad(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    z_loss: float,
    chunk_size: int,
    grad_weight: torch.Tensor,
    scale: torch.Tensor | None,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .fused_loss import _loss_grad

    w = weight.to(torch.bfloat16)
    return _loss_grad(
        hidden,
        w,
        labels,
        z_loss,
        chunk_size,
        grad_weight,
        scale,
        projection=Projection(w, precision),
    )


@loss_grad.register_fake
def _(hidden, weight, labels, z_loss, chunk_size, grad_weight, scale, precision):
    return (
        hidden.new_empty(
            (hidden.numel() // hidden.shape[-1], hidden.shape[-1]), dtype=torch.bfloat16
        ),
        hidden.new_empty((), dtype=torch.float32),
        hidden.new_empty((), dtype=torch.float32),
    )
