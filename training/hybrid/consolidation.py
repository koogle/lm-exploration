"""Benchmark candidates that retain parameter identity and optimizer grouping."""

from types import MethodType

import torch
from model_10m.model import RMSNorm, SwiGLU
from torch.nn import functional as F

from .kda import KimiDeltaAttention


def linear_group(x, weights):
    # Each weight remains its original Parameter. Concatenation and its gradient
    # split belong to this invocation, never to the optimizer's matrix grouping.
    return F.linear(x, torch.cat(weights, dim=0)).split([w.shape[0] for w in weights], dim=-1)


def projected_mlp(self, x):
    gate, up = linear_group(x, (self.gate.weight, self.up.weight))
    return self.down(F.silu(gate) * up)


def consolidated_norm(self, x):
    if x.is_cuda and x.dtype == torch.float32:
        from .kernels.rmsnorm import rms_norm

        return rms_norm(x, self.weight, self.eps)
    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
    return normalized.to(x.dtype) * self.weight.to(x.dtype)


def configure(model, candidate):
    if candidate not in ("joint-loss", "projections", "norm", "all"):
        raise ValueError(candidate)
    if model.config.looped:
        raise ValueError("Consolidation qualification targets the non-looped foundation model")
    for module in model.modules():
        if candidate in ("projections", "all"):
            if isinstance(module, KimiDeltaAttention):
                module.combined_projections = True
            if isinstance(module, SwiGLU):
                if module.gate.bias is not None or module.up.bias is not None:
                    raise ValueError("Projection qualification requires bias-free MLPs")
                module.forward = MethodType(projected_mlp, module)
        if candidate in ("norm", "all") and isinstance(module, RMSNorm):
            module.forward = MethodType(consolidated_norm, module)
