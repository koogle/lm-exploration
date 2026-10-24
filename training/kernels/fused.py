"""Autograd wrappers for the fused CUDA kernels and a helper that swaps them into a GPT.

`fuse(model)` keeps every parameter and module name, so checkpoints are interchangeable
with the unfused model. It expects BF16 autocast on CUDA, as in training.
"""

import types
from pathlib import Path

import torch
import torch.nn.functional as F

_ops = None


def ops():
    global _ops
    if _ops is None:
        from torch.utils.cpp_extension import load

        _ops = load(
            name="fused_ops",
            sources=[str(Path(__file__).with_name("fused_ops.cu"))],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _ops


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        y, rstd = ops().rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = ops().rmsnorm_backward(dy.contiguous(), x, weight, rstd)
        return dx, dw, None


class QKVRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, heads):
        return tuple(ops().qkv_rope_forward(qkv.contiguous(), heads))

    @staticmethod
    def backward(ctx, dq, dk, dv):
        return ops().qkv_rope_backward(dq, dk, dv), None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        ctx.save_for_backward(gate, up)
        return ops().swiglu_forward(gate, up)

    @staticmethod
    def backward(ctx, dh):
        gate, up = ctx.saved_tensors
        return tuple(ops().swiglu_backward(dh, gate, up))


class CrossEntropyFunction(torch.autograd.Function):
    """Summed loss over targets other than -100; the gradient is computed in the forward."""

    @staticmethod
    def forward(ctx, logits, targets):
        loss, grad = ops().cross_entropy(logits.contiguous(), targets.contiguous())
        ctx.save_for_backward(grad)
        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad.mul_(grad_output.to(grad.dtype)), None


def rmsnorm_forward(self, x):
    if x.dtype != torch.float32:
        raise ValueError("Fused RMSNorm expects the FP32 residual stream")
    return RMSNormFunction.apply(x, self.weight, self.eps)


def attention_forward(self, x):
    batch, length, width = x.shape
    query, key, value = QKVRopeFunction.apply(self.qkv(x), self.n_head)
    attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    return self.projection(attended.transpose(1, 2).contiguous().view(batch, length, width))


def swiglu_forward(self, x):
    return self.down(SwiGLUFunction.apply(self.gate(x), self.up(x)))


def gpt_forward(self, tokens, labels=None, loss_chunk_size=0):
    if loss_chunk_size:
        raise ValueError("The fused loss does not chunk")
    hidden = self.final_norm(self.blocks(self.token_embedding(tokens)))
    logits = self.lm_head(hidden)
    if labels is None:
        return logits
    return CrossEntropyFunction.apply(logits.flatten(0, 1), labels.flatten())


def fuse(model):
    """Route the GPT's element-wise work through the fused kernels, in place."""
    from model_10m.model import GPT, Attention, RMSNorm, SwiGLU

    for module in model.modules():
        for kind, forward in (
            (RMSNorm, rmsnorm_forward),
            (Attention, attention_forward),
            (SwiGLU, swiglu_forward),
            (GPT, gpt_forward),
        ):
            if isinstance(module, kind):
                module.forward = types.MethodType(forward, module)
    return model
