"""BF16 GEMM with direct accumulation into the output layer's FP32 gradient."""

import torch

from .cuda_extension import load


@torch.library.custom_op("hybrid::accumulate_output_gradient_", mutates_args=("destination",))
def accumulate_output_gradient_(
    destination: torch.Tensor, grad: torch.Tensor, hidden: torch.Tensor
) -> None:
    load("output_gradient").accumulate(destination, grad.contiguous(), hidden.contiguous())


@accumulate_output_gradient_.register_fake
def _(destination, grad, hidden):
    return None


@torch.library.custom_op(
    "hybrid::accumulate_scaled_output_gradient_", mutates_args=("destination",)
)
def accumulate_scaled_output_gradient_(
    destination: torch.Tensor, grad: torch.Tensor, hidden: torch.Tensor, coefficients: torch.Tensor
) -> None:
    """FP32-scaled GEMM accumulation; device coefficients are [alpha, beta]."""
    load("output_gradient").accumulate_scaled(
        destination, grad.contiguous(), hidden.contiguous(), coefficients
    )


@accumulate_scaled_output_gradient_.register_fake
def _(destination, grad, hidden, coefficients):
    return None
