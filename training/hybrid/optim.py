"""Muon/AdamW parameter grouping for the hybrid family.

Every parameter gets exactly one role, by name and shape:

- `muon`: 2D hidden weight matrices inside blocks and the MTP module (attention
  and KDA q/k/v/o projections, attention gate, KDA low-rank decay and output-gate
  projections, SwiGLU, MTP fuse).
- `adamw_decay`: the tied embedding/output table and the KDA beta projection
  (one row per head, not a hidden matrix).
- `adamw_no_decay`: norm scales, attention sinks, gate biases, KDA `A_log`,
  `dt_bias`, depthwise convolution kernels and any other scalar or vector.

Muon is trainer.optimizers.Muon (five-step Newton–Schulz, original aspect-ratio
scaling). Base learning rates follow the 124M Muon comparison
(distributed/MUON_COMPARISON.md); both share one schedule factor.
"""

from dataclasses import dataclass

import torch

from trainer.optimizers import Muon

__all__ = ["HybridOptimizer", "OptimizerConfig", "optimizer_role", "parameter_groups"]
ROLES = ("muon", "adamw_decay", "adamw_no_decay")


@dataclass(frozen=True)
class OptimizerConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


def optimizer_role(name, parameter):
    leaf = name.rsplit(".", 1)[-1]
    if parameter.ndim <= 1 or leaf in ("q_conv", "k_conv", "v_conv"):
        return "adamw_no_decay"
    if parameter.ndim == 2:
        if name == "token_embedding.weight" or name.endswith("mixer.b_proj.weight"):
            return "adamw_decay"
        if name.startswith(("blocks.", "mtp.")) and leaf == "weight":
            return "muon"
    raise ValueError(f"No optimizer role for {name} with shape {tuple(parameter.shape)}")


def parameter_groups(model):
    """{role: [(name, parameter)]}; tied parameters appear once."""
    groups = {role: [] for role in ROLES}
    for name, parameter in model.named_parameters():
        groups[optimizer_role(name, parameter)].append((name, parameter))
    return groups


class HybridOptimizer:
    """Muon for hidden matrices and AdamW for the rest, with one shared LR factor.

    `set_learning_rate(optimizer, factor * adamw_lr)` from trainer.optimizers
    scales both: Muon groups carry `lr_scale = muon_lr / adamw_lr`.
    """

    def __init__(self, model, config=None, *, fused=False):
        config = config or OptimizerConfig()
        groups = parameter_groups(model)
        self.muon = Muon(
            [p for _, p in groups["muon"]],
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
        )
        self.adamw = torch.optim.AdamW(
            [
                {"params": [p for _, p in groups["adamw_decay"]]},
                {"params": [p for _, p in groups["adamw_no_decay"]], "weight_decay": 0.0},
            ],
            lr=config.adamw_lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
            fused=fused,
        )
        for group in self.muon.param_groups:
            group["lr_scale"] = config.muon_lr / config.adamw_lr
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        self.base_lr = config.adamw_lr

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
