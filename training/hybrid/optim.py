"""Muon/AdamW parameter grouping for the hybrid family.

Roles are declared by the modules that create the parameters (`optimizer_roles()`
in model.py and kda.py, see roles.py):

- `muon`: 2D hidden weight matrices (attention and KDA q/k/v/o projections,
  attention gate, KDA low-rank decay and output-gate projections, SwiGLU, MTP fuse).
- `adamw_decay`: the tied embedding/output table and the KDA beta projection
  (one row per head, not a hidden matrix).
- `adamw_no_decay`: norm scales, attention sinks, gate biases, KDA `A_log`,
  `dt_bias` and depthwise convolution kernels.

Muon is trainer.optimizers.Muon (five-step Newton–Schulz, original aspect-ratio
scaling). Base learning rates follow the 124M Muon comparison
(distributed/MUON_COMPARISON.md); both share one schedule factor.
"""

from dataclasses import dataclass

import torch

from trainer.optimizers import Muon

from .roles import ROLES

__all__ = ["HybridOptimizer", "OptimizerConfig", "parameter_groups"]


@dataclass(frozen=True)
class OptimizerConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


def parameter_groups(model):
    """{role: [(name, parameter)]} from the modules' declarations; tied parameters appear once.

    Raises for a parameter that no module declares, that is declared twice, or that
    is declared `muon` without being a matrix.
    """
    declared = {}
    for module_name, module in model.named_modules():
        for parameter, role in getattr(module, "optimizer_roles", tuple)():
            if id(parameter) in declared:
                raise ValueError(f"{module_name or 'model'} declares a parameter twice")
            declared[id(parameter)] = role
    groups = {role: [] for role in ROLES}
    for name, parameter in model.named_parameters():
        role = declared.get(id(parameter))
        if role is None:
            raise ValueError(f"No module declares an optimizer role for {name}")
        if role == "muon" and parameter.ndim != 2:
            raise ValueError(f"Muon needs a matrix; {name} has shape {tuple(parameter.shape)}")
        groups[role].append((name, parameter))
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
