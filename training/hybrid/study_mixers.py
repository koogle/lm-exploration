"""Experimental KDA input mixers; production recurrence is unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from .kda import KimiDeltaAttention, kda_kernel, kda_reference, short_convolution
from .model import apply_rope


class RotatingMemory(nn.Module):
    """Exact parallel doubling scan of four damped, rotating memories."""

    def __init__(self, width):
        super().__init__()
        tau = torch.tensor([4.0, 16.0, 64.0, 256.0])
        self.log_tau = nn.Parameter(tau.log())
        self.theta = nn.Parameter(torch.tensor([0.5, 0.125, 0.03125, 0.0078125]))
        self.mix = nn.Parameter(torch.full((4, width), 0.25))

    def forward(self, u, positions):
        if u.is_cuda and getattr(self, "use_kernel", False):
            from .kernels.rotating import rotating_memory

            return rotating_memory(u, positions, self.log_tau, self.theta, self.mix)
        original_dtype = u.dtype
        z = torch.view_as_complex(u.float().reshape(*u.shape[:-1], -1, 2).contiguous())
        rho = torch.exp(-torch.exp(-self.log_tau))
        a = torch.polar(rho, self.theta)
        y = z[:, :, None, :] * (1 - rho)[None, None, :, None]
        shift = 1
        while shift < u.size(1):
            past = F.pad(y[:, :-shift], (0, 0, 0, 0, shift, 0))
            valid = (positions >= shift)[:, :, None, None]
            y = y + past * a.pow(shift)[None, None, :, None] * valid
            shift *= 2
        mixed = torch.view_as_real(y).flatten(-2)
        return (u.float() + (mixed * self.mix).sum(2)).to(original_dtype)


class StudyKDA(KimiDeltaAttention):
    def configure(self, mode, width):
        self.mode = mode
        if mode not in ("baseline", "baseline_bf16", "conv_rope"):
            for name in ("q_conv", "k_conv", "v_conv"):
                if mode != "rope_v_conv" or name != "v_conv":
                    delattr(self, name)
        if mode in ("rotating", "rotating_triton"):
            self.memories = nn.ModuleList(RotatingMemory(width) for _ in range(3))
            for memory in self.memories:
                memory.use_kernel = mode == "rotating_triton"

    def forward(self, x, rope, mask):
        shape = (*x.shape[:2], self.heads, self.head_dim)
        raw = [self.q_proj(x), self.k_proj(x), self.v_proj(x)]
        if self.mode in ("baseline", "baseline_bf16", "conv_rope"):
            values = [
                short_convolution(u, w, mask.positions)
                for u, w in zip(raw, (self.q_conv, self.k_conv, self.v_conv))
            ]
            if self.mode == "baseline_bf16":
                values = [value.to(u.dtype) for value, u in zip(values, raw)]
        elif self.mode == "rope_v_conv":
            values = [
                F.silu(raw[0]),
                F.silu(raw[1]),
                short_convolution(raw[2], self.v_conv, mask.positions).to(raw[2].dtype),
            ]
        elif self.mode in ("rotating", "rotating_triton"):
            values = [F.silu(m(u, mask.positions)) for m, u in zip(self.memories, raw)]
        else:
            values = [F.silu(u) for u in raw]
        q, k, v = [u.view(shape) for u in values]
        if self.mode in ("rope_triton", "rope_v_conv"):
            from .kernels.rope import rope_pair

            q, k = rope_pair(q, k, *rope)
        if self.mode in ("rope", "conv_rope"):
            q, k = [apply_rope(u.transpose(1, 2), *rope).transpose(1, 2) for u in (q, k)]
        args = (q, k, v, self.f_proj(x).view(shape), self.b_proj(x), self.A_log, self.dt_bias)
        out = (
            kda_kernel(*args, mask.documents) if x.is_cuda else kda_reference(*args, mask.positions)
        )
        values = out.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.norm_eps)
        values = values * self.o_norm_weight
        gate = torch.sigmoid(self.g_proj(x).view(shape).float())
        return self.o_proj((values * gate).to(x.dtype).flatten(-2))


def configure(model, variant):
    """Convert after initialization to preserve identical shared starting weights."""
    for i, block in enumerate(model.blocks):
        if block.kind != "kda":
            continue
        mode = variant
        if variant == "middle_rope":
            mode = "rope" if i % 4 == 1 else "baseline"
        if variant == "middle_rope_conv":
            mode = "conv_rope" if i % 4 == 1 else "baseline"
        block.mixer.__class__ = StudyKDA
        block.mixer.configure(mode, model.config.n_kda_head * model.config.head_dim)
    return model
