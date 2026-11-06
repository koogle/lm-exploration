"""Muon for hidden matrices and AdamW for every other parameter, by declared role.

Roles come from the modules that create the parameters (`optimizer()` methods, see
roles.py):

- `muon`: 2D hidden weight matrices.
- `adamw_decay`: AdamW with weight decay (embeddings and other non-hidden weights).
- `adamw_no_decay`: AdamW without weight decay (norm scales, sinks, gate biases, ...).

Muon is five-step quintic Newton–Schulz, following Keller Jordan's Muon reference
(https://github.com/KellerJordan/Muon). Each DDP rank updates the same reduced
gradients locally; no extra optimizer collectives or parameter sharding. Every
parameter group keeps its `base_lr`; `set_learning_rate(optimizer, factor)` applies
one schedule factor to all of them.
"""

from dataclasses import dataclass

import torch

from .roles import ROLES


def orthogonalize(update, steps=5):
    """Approximate the polar factor in BF16, orienting the shorter axis as rows."""
    x = update.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, *, lr, momentum, weight_decay):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(buffer, group["momentum"])
                update = orthogonalize(update)
                # Original Muon aspect-ratio scaling, not AdamW RMS matching.
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


@dataclass(frozen=True)
class OptimizerConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    # False trains `muon` matrices with AdamW (with weight decay) instead.
    use_muon: bool = True


def parameter_groups(model):
    """{role: [(name, parameter)]} from the modules' declarations; tied parameters appear once.

    Raises for a parameter that no module declares, that is declared twice, or that
    is declared `muon` without being a matrix.
    """
    declared = {}
    for module_name, module in model.named_modules():
        for parameter, role in getattr(module, "optimizer", tuple)():
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


class RoleOptimizer:
    """Muon for `muon` parameters and AdamW for the rest; empty groups are left out."""

    def __init__(self, model, config=None, *, fused=False):
        config = config or OptimizerConfig()
        groups = {r: [p for _, p in members] for r, members in parameter_groups(model).items()}
        if not config.use_muon:
            groups["adamw_decay"] = groups.pop("muon") + groups["adamw_decay"]
        adamw = [
            {"params": groups["adamw_decay"], "base_lr": config.adamw_lr},
            {"params": groups["adamw_no_decay"], "base_lr": config.adamw_lr, "weight_decay": 0.0},
        ]
        self.parts = {}
        if groups.get("muon"):
            self.parts["muon"] = Muon(
                [{"params": groups["muon"], "base_lr": config.muon_lr}],
                lr=config.muon_lr,
                momentum=config.muon_momentum,
                weight_decay=config.weight_decay,
            )
        self.parts["adamw"] = torch.optim.AdamW(
            [g for g in adamw if g["params"]],
            lr=config.adamw_lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
            fused=fused,
        )
        self.param_groups = [g for part in self.parts.values() for g in part.param_groups]

    @property
    def muon(self):
        return self.parts["muon"]

    @property
    def adamw(self):
        return self.parts["adamw"]

    def zero_grad(self, set_to_none=True):
        for part in self.parts.values():
            part.zero_grad(set_to_none=set_to_none)

    def step(self):
        for part in self.parts.values():
            part.step()

    def state_dict(self):
        return {name: part.state_dict() for name, part in self.parts.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.parts):
            raise ValueError("Optimizer state belongs to a different optimizer layout")
        for name, part in self.parts.items():
            part.load_state_dict(state[name])
        self.param_groups = [g for part in self.parts.values() for g in part.param_groups]


def set_learning_rate(optimizer, factor):
    """Every parameter group's learning rate to `factor` times its base rate."""
    for group in optimizer.param_groups:
        group["lr"] = factor * group["base_lr"]


def make_optimizer(model, config, *, fused=True):
    """The optimizer a subword `DistributedConfig` names: Muon plus AdamW, or AdamW only."""
    settings = OptimizerConfig(
        muon_lr=config.muon_lr,
        muon_momentum=config.muon_momentum,
        adamw_lr=config.peak_lr,
        adam_betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        use_muon=config.optimizer == "muon",
    )
    return RoleOptimizer(model, settings, fused=fused)
