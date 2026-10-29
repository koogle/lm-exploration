"""Autograd wrappers for the fused CUDA kernels and a fused GPT forward.

`fuse(model)` keeps every parameter and module name, so checkpoints are interchangeable
with the unfused model. It expects BF16 autocast on CUDA, as in training. In the fused
forward every linear runs without its bias; the kernel that consumes its output adds the
bias and returns the bias gradient, so no separate bias reductions run:

    x0 = embed(tokens)                                # FP32 residual stream
    y  = RMSNorm(x0)
    per block:
        a       = projection(attention(QKV+RoPE(qkv(y) + b_qkv)))   # per-document RoPE and attention
        x, y    = AddRMSNorm(x, a + b_projection)                    # residual add + next norm
        m       = down(SwiGLU(gate(y) + b_gate, up(y) + b_up))
        x, y    = AddRMSNorm(x, m + b_down)
    loss = LinearCrossEntropy(y, lm_head), chunked so full logits never exist
"""

import types
from pathlib import Path

import torch
import torch.nn.functional as F
from model_10m.model import document_attention, find_documents

_ops = None
LOSS_CHUNK_ROWS = 4096


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
    """FP32 input, BF16 output."""

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


class AddRMSNormFunction(torch.autograd.Function):
    """Return (residual + delta + bias in FP32, its RMSNorm in BF16)."""

    @staticmethod
    def forward(ctx, residual, delta, bias, weight, eps):
        total, y, rstd = ops().add_rmsnorm_forward(
            residual.contiguous(), delta.contiguous(), bias, weight, eps
        )
        ctx.save_for_backward(total, weight, rstd)
        return total, y

    @staticmethod
    def backward(ctx, d_total, dy):
        total, weight, rstd = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros(total.shape, device=total.device, dtype=torch.bfloat16)
        d_residual, d_delta, dw, db = ops().add_rmsnorm_backward(
            dy.contiguous(), None if d_total is None else d_total.contiguous(), total, weight, rstd
        )
        return d_residual, d_delta, db, dw, None


class QKVRopeFunction(torch.autograd.Function):
    """q, k, v in [B, H, T, D]; `positions` [B, T] are positions within each document."""

    @staticmethod
    def forward(ctx, qkv, bias, positions, heads):
        ctx.save_for_backward(positions)
        return tuple(ops().qkv_rope_forward(qkv.contiguous(), bias, positions, heads))

    @staticmethod
    def backward(ctx, dq, dk, dv):
        (positions,) = ctx.saved_tensors
        dqkv, dbias = ops().qkv_rope_backward(dq, dk, dv, positions)
        return dqkv, dbias, None, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, gate_bias, up_bias):
        gate, up = gate.contiguous(), up.contiguous()
        ctx.save_for_backward(gate, up, gate_bias, up_bias)
        return ops().swiglu_forward(gate, up, gate_bias, up_bias)

    @staticmethod
    def backward(ctx, dh):
        gate, up, gate_bias, up_bias = ctx.saved_tensors
        return tuple(ops().swiglu_backward(dh, gate, up, gate_bias, up_bias))


class LinearCrossEntropyFunction(torch.autograd.Function):
    """Summed cross-entropy of hidden @ weight.T over targets other than -100.

    Rows are processed in chunks: logits, the loss and their gradient (written over the
    logits), then the hidden and weight gradients. Only one chunk's logits exist at a time.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        weight16 = weight.to(torch.bfloat16)
        d_hidden = torch.empty_like(hidden)
        d_weight = torch.zeros_like(weight)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.size(0), LOSS_CHUNK_ROWS):
            rows = slice(start, start + LOSS_CHUNK_ROWS)
            logits = hidden[rows] @ weight16.t()
            total += ops().cross_entropy_(logits, targets[rows]).sum()
            d_hidden[rows] = logits @ weight16
            d_weight += logits.t() @ hidden[rows]
        ctx.save_for_backward(d_hidden, d_weight)
        return total

    @staticmethod
    def backward(ctx, grad_output):
        d_hidden, d_weight = ctx.saved_tensors
        return d_hidden * grad_output.to(d_hidden.dtype), d_weight * grad_output, None


def attention(module, y, documents):
    batch, length, width = y.shape
    query, key, value = QKVRopeFunction.apply(
        F.linear(y, module.qkv.weight), module.qkv.bias, documents.positions, module.n_head
    )
    attended = document_attention(*(t.transpose(1, 2) for t in (query, key, value)), documents)
    return F.linear(attended.reshape(batch, length, width), module.projection.weight)


def mlp(module, y):
    hidden = SwiGLUFunction.apply(
        F.linear(y, module.gate.weight),
        F.linear(y, module.up.weight),
        module.gate.bias,
        module.up.bias,
    )
    return F.linear(hidden, module.down.weight)


def gpt_forward(self, tokens, labels=None, loss_chunk_size=0):
    if loss_chunk_size:
        raise ValueError("The fused loss always chunks; loss_chunk_size is not used")
    documents = find_documents(tokens, self.config.eos_id)
    blocks = list(self.blocks)
    norms = [b.mlp_norm for b in blocks]
    following = [b.attention_norm for b in blocks[1:]] + [self.final_norm]
    x = self.token_embedding(tokens)
    y = RMSNormFunction.apply(x, blocks[0].attention_norm.weight, blocks[0].attention_norm.eps)
    for block, mlp_norm, next_norm in zip(blocks, norms, following):
        a = attention(block.attention, y, documents)
        x, y = AddRMSNormFunction.apply(
            x, a, block.attention.projection.bias, mlp_norm.weight, mlp_norm.eps
        )
        m = mlp(block.mlp, y)
        x, y = AddRMSNormFunction.apply(x, m, block.mlp.down.bias, next_norm.weight, next_norm.eps)
    if labels is None:
        return F.linear(y, self.lm_head.weight)
    return LinearCrossEntropyFunction.apply(y.flatten(0, 1), self.lm_head.weight, labels.flatten())


def fuse(model):
    """Route a GPT's forward through the fused kernels, in place.

    Attention is the model's own document-causal FlashAttention varlen call.
    """
    model.forward = types.MethodType(gpt_forward, model)
    return model
