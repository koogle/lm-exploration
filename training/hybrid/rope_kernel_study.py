"""Numerical and training checks for fused KDA RoPE."""

import time

import torch

from .kernels.rope import rope_pair
from .model import HybridConfig, HybridLM, apply_rope, rope_tables
from .study_mixers import configure


def qualify():
    results = []
    for dtype in [torch.float32, torch.bfloat16]:
        for batch, length, heads, width in [(2, 1, 2, 16), (2, 17, 2, 64), (8, 1024, 12, 64)]:
            torch.manual_seed(512)
            q, k = [
                torch.randn(
                    batch, length, heads, width, device="cuda", dtype=dtype, requires_grad=True
                )
                for _ in range(2)
            ]
            a, b = [x.detach().clone().requires_grad_() for x in (q, k)]
            tables = rope_tables(length, width, 1e6, "cuda")
            positions = torch.arange(length, device="cuda").repeat(batch, 1) % max(1, length // 3)
            cos, sin = [t[positions][:, None] for t in tables]
            ref = [apply_rope(x.transpose(1, 2), cos, sin).transpose(1, 2) for x in [q, k]]
            fast = rope_pair(a, b, cos, sin)
            grad = [torch.randn_like(x) for x in ref]
            gr = torch.autograd.grad(ref, (q, k), grad)
            gf = torch.autograd.grad(fast, (a, b), grad)
            metrics = {}
            for name, x, y in zip(["q", "k", "dq", "dk"], [*ref, *gr], [*fast, *gf]):
                absolute = (x.float() - y.float()).abs().max().item()
                relative = (
                    (x.float() - y.float()).norm() / x.float().norm().clamp_min(1e-8)
                ).item()
                assert relative < 1e-6 or absolute < 1e-7, (dtype, length, name, relative, absolute)
                metrics[name] = {"max_abs": absolute, "relative_l2": relative}
            results.append({"dtype": str(dtype), "shape": list(q.shape), "errors": metrics})
    return results


def compare_training(arr, config, deadline):
    import math

    import numpy as np

    models = {}
    for variant in ["rope", "rope_triton"]:
        torch.manual_seed(17)
        model = configure(HybridLM(HybridConfig(**config)), variant).cuda()
        opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        models[variant] = model, opt
    records = []
    for step in range(100):
        assert time.time() < deadline - 120
        start = step * 8192
        rows = np.stack([arr[start + i * 1024 : start + (i + 1) * 1024 + 1] for i in range(8)])
        x = torch.tensor(rows[:, :-1], device="cuda", dtype=torch.long)
        y = torch.tensor(rows[:, 1:], device="cuda", dtype=torch.long)
        y[y == 1] = -100
        losses = {}
        order = list(models)
        if step % 2:
            order.reverse()
        for variant in order:
            model, opt = models[variant]
            for group in opt.param_groups:
                group["lr"] = (
                    3e-4
                    * min(1.0, (step + 1) / 400)
                    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / 12288)))
                )
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, y)
                loss = out["loss"] / out["count"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            losses[variant] = (out["ce"] / out["count"]).item()
        records.append({"step": step + 1, **losses})
        if (step + 1) % 20 == 0:
            print("RoPE matched training", records[-1], flush=True)
    return {
        "curve": records,
        "max_ce_difference": max(abs(r["rope"] - r["rope_triton"]) for r in records),
    }
