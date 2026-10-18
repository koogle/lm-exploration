"""Muon for hidden matrices with an AdamW companion for all other parameters.

Five-step quintic Newton–Schulz Muon, following Keller Jordan's Muon reference
(https://github.com/KellerJordan/Muon). Each DDP rank updates the same reduced
gradients locally; no extra optimizer collectives or parameter sharding.
"""

import torch


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


class HybridMuon:
    def __init__(self, model, config, *, fused):
        hidden, auxiliary = [], []
        for name, parameter in model.named_parameters():
            (hidden if name.startswith("blocks.") and parameter.ndim == 2 else auxiliary).append(
                parameter
            )
        if not hidden or not auxiliary:
            raise ValueError("Muon requires hidden matrices and auxiliary parameters")
        self.muon = Muon(
            hidden,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
        )
        self.adamw = torch.optim.AdamW(
            auxiliary,
            lr=config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=fused,
        )
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        for group in self.muon.param_groups:
            group["lr_scale"] = config.muon_lr / config.peak_lr

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


def make_optimizer(model, config, *, fused=True):
    if config.optimizer == "muon":
        return HybridMuon(model, config, fused=fused)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=fused,
    )


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
