"""Numerical qualification and same-device system measurements for every candidate."""

import copy
import time

import torch

from .autoresearch_model import build, state_bytes
from .autoresearch_specs import PROXY, TARGET
from .kda import short_convolution
from .model import DocumentMask


def discrepancy(a, b):
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-8)).item(),
    }


def check_conv():
    from .kernels.local_conv import local_conv

    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        for length, kernel in ((1, 4), (17, 4), (129, 8), (129, 16)):
            x = torch.randn(2, length, 64, device="cuda", dtype=dtype, requires_grad=True)
            w = torch.randn(64, 1, kernel, device="cuda", requires_grad=True) * 0.2
            p = (torch.arange(length, device="cuda") % 13).repeat(2, 1)
            xr = x.detach().clone().requires_grad_()
            wr = w.detach().clone().requires_grad_()
            a, b = local_conv(x, w, p), short_convolution(xr, wr, p)
            dy = torch.randn_like(a)
            ga = torch.autograd.grad(a, (x, w), dy)
            gb = torch.autograd.grad(b, (xr, wr), dy)
            errors = [discrepancy(u, v) for u, v in zip((a, *ga), (b, *gb))]
            for e in errors:
                assert (
                    e["relative_l2"] < (0.015 if dtype == torch.bfloat16 else 3e-5)
                    or e["max_abs"] < 2e-5
                ), e
            checks.append(
                {"dtype": str(dtype), "length": length, "kernel": kernel, "errors": errors}
            )
    return checks


def qualify(variant):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    cfg = dict(
        PROXY,
        vocab_size=512,
        eos_id=511,
        n_embd=128,
        n_head=4,
        n_kv_head=2,
        n_kda_head=4,
        head_dim=32,
        mlp_hidden=256,
        block_size=128,
    )
    model = build(cfg, variant, 17)
    layer = next(b.mixer for b in model.blocks if b.kind == "kda").cuda()
    ref = copy.deepcopy(layer)
    ref.reference = True
    docs = (torch.arange(17, device="cuda") // 7).repeat(2, 1)
    mask = DocumentMask(docs)
    rope = (
        model.rope_cos.cuda()[mask.positions][:, None],
        model.rope_sin.cuda()[mask.positions][:, None],
    )
    x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, yr = layer(x, rope, mask), ref(xr, rope, mask)
    dy = torch.randn_like(y)
    ga = torch.autograd.grad(y, (x, *layer.parameters()), dy, allow_unused=True)
    gb = torch.autograd.grad(yr, (xr, *ref.parameters()), dy, allow_unused=True)
    errors = []
    for name, a, b in zip(
        ["output", "input", *dict(layer.named_parameters())], (y, *ga), (yr, *gb)
    ):
        assert (a is None) == (b is None), name
        if a is None:
            continue
        e = discrepancy(a, b)
        assert e["relative_l2"] < 0.025 or e["max_abs"] < 5e-5, (name, e)
        assert torch.isfinite(a).all(), name
        errors.append({"tensor": name, **e})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        changed = x.detach().clone()
        changed[:, :7] += 5
        assert torch.equal(layer(changed, rope, mask)[:, 7:], y[:, 7:]), "document leakage"
        changed = x.detach().clone()
        changed[:, 10:] += 5
        assert torch.equal(layer(changed, rope, mask)[:, :10], y[:, :10]), "future leakage"
    preserved = None
    if variant in ("baseline", "fused", "kv8", "kv16", "dynamic_kv", "input_gate", "memory_gate"):
        base = build(cfg, "baseline", 17).blocks[0].mixer.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = base(x, rope, mask)
        preserved = discrepancy(y, expected)
        assert preserved["relative_l2"] < 0.005 or preserved["max_abs"] < 5e-5, preserved
    result = {
        "variant": variant,
        "errors": errors,
        "causal": True,
        "document_reset": True,
        "initial_baseline_function": preserved,
    }
    if variant == "fused":
        result["conv_checks"] = check_conv()
    return result


def benchmark(variant, length=256):
    """Matched layer benchmark excludes first-call compilation and optimizer."""
    cfg = dict(PROXY, block_size=max(4096, length))
    # Construct only a single mixer-sized model to avoid needless embedding allocations.
    small = dict(cfg, vocab_size=512, eos_id=511, n_layer=12)
    measurements = {}
    for name in ("baseline", variant):
        torch.manual_seed(19)
        model = build(small, name, 19)
        layer = next(b.mixer for b in model.blocks if b.kind == "kda").cuda()
        docs = torch.zeros((1, length), device="cuda", dtype=torch.long)
        mask = DocumentMask(docs)
        rope = (
            model.rope_cos.cuda()[mask.positions][:, None],
            model.rope_sin.cuda()[mask.positions][:, None],
        )
        x = torch.randn(1, length, cfg["n_embd"], device="cuda", requires_grad=True)
        elapsed = []
        for i in range(4):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = layer(x, rope, mask)
            y.float().square().mean().backward()
            torch.cuda.synchronize()
            if i:
                elapsed.append(time.perf_counter() - start)
        measurements[name] = {
            "seconds": sum(elapsed) / len(elapsed),
            "parameters": sum(p.numel() for p in layer.parameters()),
            "state_bytes": layer.heads * layer.key_dim * layer.value_dim * 4,
        }
        del model, layer, x, y
        torch.cuda.empty_cache()
    ratio = measurements[variant]["seconds"] / measurements["baseline"]["seconds"]
    return {"length": length, "measurements": measurements, "slowdown": ratio}


def counts(variant):
    result = {}
    # Meta construction avoids allocating a target-scale model just for accounting.
    for name, cfg in [("proxy", PROXY), ("target", TARGET)]:
        with torch.device("meta"):
            model = build(cfg, variant, 17)
        total = sum(p.numel() for p in model.parameters())
        result[name] = {
            "parameters": total,
            "nonembedding_parameters": total - cfg["vocab_size"] * cfg["n_embd"],
            "state_bytes_per_sequence": state_bytes(model),
            "full_layers": sum(b.kind == "full" for b in model.blocks),
        }
    return result
