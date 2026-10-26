"""Numerical and matched-training qualification for the rotating Triton kernels."""

import time

import torch

from .model import HybridConfig, HybridLM
from .study_mixers import RotatingMemory, configure


def qualify():
    errors = []
    for dtype in [torch.float32, torch.bfloat16]:
        for length in [1, 17, 128, 512, 1024]:
            torch.manual_seed(421)
            ref = RotatingMemory(64).cuda()
            custom = RotatingMemory(64).cuda()
            custom.load_state_dict(ref.state_dict())
            custom.use_kernel = True
            u = torch.randn(2, length, 64, device="cuda", dtype=dtype, requires_grad=True)
            v = u.detach().clone().requires_grad_()
            pos = torch.arange(length, device="cuda").repeat(2, 1)
            if length > 2:
                split = length // 3
                pos[:, split:] -= split
            y = ref(u, pos)
            z = custom(v, pos)
            grad = torch.randn_like(y)
            a = torch.autograd.grad(y, (u, *ref.parameters()), grad, allow_unused=True)
            b = torch.autograd.grad(z, (v, *custom.parameters()), grad, allow_unused=True)
            a = [torch.zeros_like(p) if g is None else g for p, g in zip((u, *ref.parameters()), a)]
            b = [
                torch.zeros_like(p) if g is None else g
                for p, g in zip((v, *custom.parameters()), b)
            ]
            pairs = [
                ("output", y, z),
                *[(name, x, y) for name, x, y in zip(["input", "log_tau", "theta", "mix"], a, b)],
            ]
            metrics = {}
            for name, x, y in pairs:
                absolute = (x.float() - y.float()).abs().max().item()
                relative = (
                    (x.float() - y.float()).norm() / x.float().norm().clamp_min(1e-8)
                ).item()
                metrics[name] = {"max_abs": absolute, "relative_l2": relative}
                threshold = 0.0002 if dtype == torch.bfloat16 else 0.00003
                assert relative < threshold or absolute < 2e-5, (dtype, length, name, metrics[name])
            errors.append({"dtype": str(dtype), "length": length, "errors": metrics})
            print("Kernel check", dtype, length, metrics, flush=True)
    return errors


def compare_training(arr, deadline):
    import numpy as np

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = HybridConfig(
        vocab_size=32768,
        eos_id=2,
        block_size=512,
        n_layer=8,
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        n_kda_head=4,
        head_dim=64,
        mlp_hidden=768,
        loss_chunk_size=1024,
    )
    models = {}
    for variant in ["rotating", "rotating_triton"]:
        torch.manual_seed(17)
        model = configure(HybridLM(cfg), variant).cuda()
        opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        models[variant] = (model, opt)
    records = []
    times = {v: [] for v in models}
    import math

    for step in range(100):
        assert time.time() < deadline - 60
        start = step * 4096
        rows = np.stack([arr[start + i * 512 : start + (i + 1) * 512 + 1] for i in range(8)])
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
                    * min(1.0, (step + 1) / 50)
                    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / 800)))
                )
            torch.cuda.synchronize()
            tick = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, y)
                loss = out["loss"] / out["count"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            if step >= 10:
                times[variant].append(time.perf_counter() - tick)
            losses[variant] = (out["ce"] / out["count"]).item()
        records.append({"step": step + 1, **losses})
        if (step + 1) % 20 == 0:
            print("Matched training", records[-1], flush=True)
    differences = [abs(r["rotating"] - r["rotating_triton"]) for r in records]
    return {
        "curve": records,
        "seconds_per_step": times,
        "max_ce_difference": max(differences),
        "gpu": torch.cuda.get_device_name(),
    }
