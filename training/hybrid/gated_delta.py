"""Benchmark-only scalar-decay Gated DeltaNet comparison using shared KDA projections.

Replaces per-key-channel forgetting by one decay scalar per head. This changes
model capacity; it is an architecture comparison, not an equivalent KDA kernel.
"""

import math

import torch
from torch import nn

from trainer.roles import declare

from .documents import cu_seqlens
from .kda import KimiDeltaAttention, kda_reference


def scalar_reference(q, k, v, g, beta, A_log, dt_bias, positions, final_state=False):
    width = q.shape[-1]
    return kda_reference(
        q,
        k,
        v,
        g.expand(*g.shape[:-1], width),
        beta,
        A_log,
        dt_bias[:, None].expand(-1, width).reshape(-1),
        positions,
        final_state,
    )


def scalar_kernel(q, k, v, g, beta, A_log, dt_bias, documents, final_state=False):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    batch, length = documents.shape
    out, state = chunk_gated_delta_rule(
        *(t.reshape(1, batch * length, *t.shape[2:]) for t in (q, k, v)),
        g=g.squeeze(-1).reshape(1, batch * length, q.shape[2]),
        beta=beta.reshape(1, batch * length, q.shape[2]),
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        cu_seqlens=cu_seqlens(documents).long(),
        output_final_state=final_state,
    )
    out = out.view(batch, length, *out.shape[2:])
    return (out, state) if final_state else out


class GatedDeltaAttention(KimiDeltaAttention):
    recurrence_kernel = staticmethod(scalar_kernel)
    recurrence_reference = staticmethod(scalar_reference)

    def __init__(self, config):
        super().__init__(config)
        self.f_proj = nn.Linear(config.n_embd, self.heads, bias=False)
        dt = torch.exp(torch.rand(self.heads) * math.log(100) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def optimizer(self):
        yield from declare("muon", self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj)
        yield from declare("adamw_decay", self.b_proj, self.f_proj)
        yield from declare(
            "adamw_no_decay",
            self.q_conv,
            self.k_conv,
            self.v_conv,
            self.A_log,
            self.dt_bias,
            self.o_norm_weight,
        )
        self.o_proj.residual_output = True


def configure(model):
    """Replace KDA blocks before optimizer construction; keep full-attention layers."""
    if model.config.looped:
        raise ValueError("Scalar-delta qualification targets the non-looped foundation model")
    for block in model.blocks:
        if isinstance(block.mixer, KimiDeltaAttention):
            old = block.mixer
            replacement = GatedDeltaAttention(model.config).to(old.q_proj.weight.device)
            # Preserve common weights so differences are restricted to the gate.
            common = {
                k: v
                for k, v in old.state_dict().items()
                if k in replacement.state_dict() and replacement.state_dict()[k].shape == v.shape
            }
            replacement.load_state_dict(common, strict=False)
            nn.init.normal_(replacement.f_proj.weight, std=model.config.init_std)
            block.mixer = replacement
