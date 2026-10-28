"""Registered, one-mechanism KDA candidates; every arm retains local q/k/v features."""

import copy
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .autoresearch_specs import VARIANTS
from .kda import KimiDeltaAttention, kda_gate, kda_kernel, kda_reference, short_convolution
from .model import HybridConfig, HybridLM


def feedback_recurrence(q, k, v, g, beta, A_log, dt_bias, positions, feedback):
    """Nonlinear memory-dependent write gate, deliberately explicit reference.

    This cannot be passed to the affine chunked KDA kernel. Qualification must
    measure its sequential cost and may reject it before expensive training.
    """
    q = F.normalize(q.float(), dim=-1) * q.size(-1) ** -0.5
    k, v = F.normalize(k.float(), dim=-1), v.float()
    decay = kda_gate(g, A_log, dt_bias).exp()
    state = q.new_zeros(q.size(0), q.size(2), q.size(3), v.size(-1))
    outputs = []
    for t in range(q.size(1)):
        state = state * (positions[:, t] > 0)[:, None, None, None] * decay[:, t, ..., None]
        read = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        error = v[:, t] - read
        scale = v[:, t].square().mean(-1).add(1e-5)
        novelty = torch.log1p(error.square().mean(-1) / scale)
        occupancy = torch.log1p(read.square().mean(-1) / scale)
        b = (beta[:, t].float() + feedback[:, 0] * novelty + feedback[:, 1] * occupancy).sigmoid()
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t] * b[..., None], error)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outputs, 1)


class ResearchKDA(KimiDeltaAttention):
    def setup(self, config, variant):
        self.variant = variant
        self.key_dim = self.head_dim
        self.value_dim = self.head_dim
        self.banks = 1
        self.use_fused = variant == "fused"
        width = self.heads * self.head_dim
        if variant in ("banks2", "slow_bank", "key_wide", "heads_narrow"):
            self.original_heads = self.heads
            if variant in ("banks2", "slow_bank"):
                self.banks = 2
                self.heads *= 2
                if variant == "banks2":
                    self.key_dim //= 2
            elif variant == "key_wide":
                self.key_dim *= 2
            else:
                self.heads *= 2
                self.key_dim //= 2
                self.value_dim //= 2
            inner = self.heads * self.key_dim
            old = copy.deepcopy(self.state_dict())
            self.q_proj = nn.Linear(config.n_embd, inner, bias=False)
            self.k_proj = nn.Linear(config.n_embd, inner, bias=False)
            self.q_conv = nn.Parameter(torch.empty(inner, 1, 4).uniform_(-0.5, 0.5))
            self.k_conv = nn.Parameter(torch.empty(inner, 1, 4).uniform_(-0.5, 0.5))
            self.f_proj = nn.Sequential(
                nn.Linear(config.n_embd, self.key_dim, bias=False),
                nn.Linear(self.key_dim, inner, bias=False),
            )
            self.b_proj = nn.Linear(config.n_embd, self.heads, bias=False)
            self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
            dt = torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)).exp()
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            self.o_norm_weight = nn.Parameter(torch.ones(self.value_dim))
            for module in (self.q_proj, self.k_proj, self.f_proj, self.b_proj):
                module.apply(
                    lambda m: (
                        nn.init.normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else None
                    )
                )
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if name in old and old[name].shape == p.shape:
                        p.copy_(old[name])
        if variant in ("head_timescales", "slow_bank"):
            tau = torch.logspace(math.log10(8), math.log10(8192), self.heads)
            if variant == "slow_bank":
                tau = torch.tensor([16.0, 4096.0]).repeat(self.original_heads)
            dt = (1 / (tau * self.A_log.detach().exp()))[:, None].expand(-1, self.key_dim).flatten()
            with torch.no_grad():
                self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if variant in ("kv8", "kv16"):
            length = int(variant[2:])
            for name in ("k_conv", "v_conv"):
                old = getattr(self, name)
                new = old.new_zeros(old.size(0), 1, length)
                new[:, :, -4:] = old.detach()
                setattr(self, name, nn.Parameter(new))
        if variant == "dynamic_kv":
            # Signed, input-dependent mixture of a second local filter. Zero
            # filters preserve the base function while the gate is nonconstant.
            self.dynamic_filters = nn.Parameter(torch.zeros(2, width, 1, 8))
            self.filter_gate = nn.Linear(config.n_embd, 2 * self.heads, bias=False)
            nn.init.normal_(self.filter_gate.weight, std=0.02)
        if variant == "input_gate":
            self.write_gate = nn.Sequential(
                nn.Linear(config.n_embd, 32, bias=False),
                nn.SiLU(),
                nn.Linear(32, self.heads, bias=False),
            )
            nn.init.normal_(self.write_gate[0].weight, std=0.02)
            nn.init.zeros_(self.write_gate[2].weight)
        if variant == "memory_gate":
            self.feedback = nn.Parameter(torch.zeros(self.heads, 2))
        self.reference = False

    def forward(self, x, rope, mask):
        b, t, _ = x.shape
        conv = short_convolution
        if self.use_fused and x.is_cuda and not self.reference:
            from .kernels.local_conv import local_conv

            conv = local_conv
        raw = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
        q, k, v = [
            conv(u, w, mask.positions) for u, w in zip(raw, (self.q_conv, self.k_conv, self.v_conv))
        ]
        if self.variant == "dynamic_kv":
            gates = self.filter_gate(x).float().view(b, t, 2, self.heads).tanh()
            extras = [
                short_convolution(u, w, mask.positions).view(b, t, self.heads, -1)
                for u, w in zip(raw[1:], self.dynamic_filters)
            ]
            k = (k.view(b, t, self.heads, -1) + gates[:, :, 0, :, None] * extras[0]).flatten(-2)
            v = (v.view(b, t, self.heads, -1) + gates[:, :, 1, :, None] * extras[1]).flatten(-2)
        q, k = [u.view(b, t, self.heads, self.key_dim) for u in (q, k)]
        if self.banks == 2:
            v = v.view(b, t, self.original_heads, self.value_dim).repeat_interleave(2, dim=2)
        else:
            v = v.view(b, t, self.heads, self.value_dim)
        g, beta = self.f_proj(x).view(b, t, self.heads, self.key_dim), self.b_proj(x)
        if self.variant == "input_gate":
            beta = beta + self.write_gate(x)
        args = (q, k, v, g, beta, self.A_log, self.dt_bias)
        if self.variant == "memory_gate":
            out = feedback_recurrence(*args, mask.positions, self.feedback)
        elif self.reference or not x.is_cuda:
            out = kda_reference(*args, mask.positions)
        else:
            out = kda_kernel(*args, mask.documents)
        if self.banks == 2:
            out = out.view(b, t, self.original_heads, 2, self.value_dim).mean(3)
        values = out.float()
        values = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.norm_eps)
            * self.o_norm_weight
        )
        gate = self.g_proj(x).float().view_as(values).sigmoid()
        return self.o_proj((values * gate).to(x.dtype).flatten(-2))


def build(config, variant, seed):
    if variant not in VARIANTS:
        raise ValueError(variant)
    cfg = HybridConfig(**config)
    if variant == "layout_dense":
        cfg = replace(cfg, layer_pattern=("kda", "kda", "full"))
    if variant == "layout_sparse":
        cfg = replace(cfg, layer_pattern=("kda",) * 5 + ("full",))
    if variant == "layout_front":
        cfg = replace(cfg, layer_pattern=("kda", "full", "kda", "kda"))
    torch.manual_seed(seed)
    model = HybridLM(cfg)
    # Preserve shape-compatible common weights across layout variants too.
    if variant.startswith("layout_"):
        torch.manual_seed(seed)
        base = HybridLM(HybridConfig(**config)).state_dict()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in base and p.shape == base[n].shape:
                    p.copy_(base[n])
    for block in model.blocks:
        if block.kind == "kda":
            block.mixer.__class__ = ResearchKDA
            block.mixer.setup(cfg, variant)
    return model


def state_bytes(model):
    return sum(
        b.mixer.heads * b.mixer.key_dim * b.mixer.value_dim * 4
        for b in model.blocks
        if b.kind == "kda"
    )
