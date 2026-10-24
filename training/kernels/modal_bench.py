"""Check the fused kernels against PyTorch and time 124M training steps on an H100.

Run from training/:
    modal run -m kernels.modal_bench [--output artifacts/kernels/bench.json]
"""

import json
import time
from pathlib import Path

import modal

from kernels.image import LAUNCH_LIMIT_SECONDS, image, limits

app = modal.App("kernel-bench")


def relative_error(actual, expected):
    return float(
        (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    )


def op_checks():
    import torch
    import torch.nn.functional as F
    from model_10m.model import rotary

    from kernels import fused

    torch.manual_seed(0)
    results = {}

    def compare(actual, expected):
        return {
            name: relative_error(a, e)
            for name, (a, e) in zip(expected, zip(actual, expected.values()))
        }

    # RMSNorm: FP32 residual in, BF16 out.
    x = torch.randn(4096, 768, device="cuda", requires_grad=True)
    w = (1 + 0.1 * torch.randn(768, device="cuda")).requires_grad_()
    dy = torch.randn(4096, 768, device="cuda").bfloat16()
    reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w
    expected = dict(
        zip(
            ("forward", "dx", "dw"),
            (reference, *torch.autograd.grad(reference, (x, w), dy.float())),
        )
    )
    y = fused.RMSNormFunction.apply(x, w, 1e-5)
    results["rmsnorm"] = compare((y, *torch.autograd.grad(y, (x, w), dy)), expected)

    # Residual add + RMSNorm with the delta's bias.
    r = torch.randn(4096, 768, device="cuda", requires_grad=True)
    d = torch.randn(4096, 768, device="cuda").bfloat16().requires_grad_()
    b = (0.1 * torch.randn(768, device="cuda")).requires_grad_()
    d_total = torch.randn(4096, 768, device="cuda")
    total = r + d.float() + b
    normed = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-5) * w
    grads = torch.autograd.grad((total, normed), (r, d, b, w), (d_total, dy.float()))
    expected = dict(zip(("total", "normed", "dr", "dd", "db", "dw"), (total, normed, *grads)))
    t, n = fused.AddRMSNormFunction.apply(r, d, b, w, 1e-5)
    results["add_rmsnorm"] = compare(
        (t, n, *torch.autograd.grad((t, n), (r, d, b, w), (d_total, dy))), expected
    )

    # QKV split + RoPE with bias, against the model's own rotary().
    qkv = torch.randn(2, 512, 3 * 768, device="cuda").bfloat16().requires_grad_()
    bias = (0.1 * torch.randn(3 * 768, device="cuda")).requires_grad_()
    heads = 12

    def reference_qkv(t, bias):
        q, k, v = (t.float() + bias).chunk(3, -1)
        q, k, v = [z.view(2, 512, heads, 64).transpose(1, 2) for z in (q, k, v)]
        return rotary(q), rotary(k), v

    outputs = [torch.randn(2, heads, 512, 64, device="cuda").bfloat16() for _ in range(3)]
    reference = reference_qkv(qkv, bias)
    grads = torch.autograd.grad(reference, (qkv, bias), [g.float() for g in outputs])
    expected = dict(zip(("q", "k", "v", "dqkv", "dbias"), (*reference, *grads)))
    actual = fused.QKVRopeFunction.apply(qkv, bias, heads)
    results["qkv_rope"] = compare(
        (*actual, *torch.autograd.grad(actual, (qkv, bias), outputs)), expected
    )

    # SwiGLU activation with both biases.
    g = torch.randn(8192, 2048, device="cuda").bfloat16().requires_grad_()
    u = torch.randn(8192, 2048, device="cuda").bfloat16().requires_grad_()
    bg = (0.1 * torch.randn(2048, device="cuda")).requires_grad_()
    bu = (0.1 * torch.randn(2048, device="cuda")).requires_grad_()
    dh = torch.randn(8192, 2048, device="cuda").bfloat16()
    reference = F.silu(g.float() + bg) * (u.float() + bu)
    grads = torch.autograd.grad(reference, (g, u, bg, bu), dh.float())
    expected = dict(zip(("forward", "dgate", "dup", "dgate_bias", "dup_bias"), (reference, *grads)))
    h = fused.SwiGLUFunction.apply(g, u, bg, bu)
    results["swiglu"] = compare((h, *torch.autograd.grad(h, (g, u, bg, bu), dh)), expected)

    # LM head + cross-entropy in chunks, with ignored targets.
    hidden = torch.randn(10000, 768, device="cuda").bfloat16().requires_grad_()
    weight = (0.02 * torch.randn(32768, 768, device="cuda")).requires_grad_()
    targets = torch.randint(4, 32768, (10000,), device="cuda")
    targets[::7] = -100
    logits = hidden.float() @ weight.t()
    reference = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    expected = dict(
        zip(
            ("loss", "dhidden", "dweight"),
            (reference, *torch.autograd.grad(reference, (hidden, weight))),
        )
    )
    loss = fused.LinearCrossEntropyFunction.apply(hidden, weight, targets)
    results["linear_cross_entropy"] = compare(
        (loss, *torch.autograd.grad(loss, (hidden, weight))), expected
    )
    return results


def model_check():
    """Loss and gradients of the whole model, fused versus unfused, same weights."""
    import torch
    from model_100m.model import GPT, ModelConfig

    from kernels import fused

    torch.manual_seed(0)
    config = ModelConfig(vocab_size=32768, block_size=2048)
    reference = GPT(config).cuda()
    candidate = GPT(config).cuda()
    candidate.load_state_dict(reference.state_dict())
    fused.fuse(candidate)
    tokens = torch.randint(4, 32768, (4, 2048), device="cuda")
    labels = torch.randint(4, 32768, (4, 2048), device="cuda")
    labels[:, :17] = -100
    out = {}
    for name, model in (("reference", reference), ("fused", candidate)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, labels) / (labels != -100).sum()
        loss.backward()
        out[name] = (float(loss), torch.cat([p.grad.flatten() for p in model.parameters()]))
    cosine = torch.nn.functional.cosine_similarity(out["fused"][1], out["reference"][1], dim=0)
    return {
        "reference_loss": out["reference"][0],
        "fused_loss": out["fused"][0],
        "gradient_cosine": float(cosine),
        "gradient_relative_error": relative_error(out["fused"][1], out["reference"][1]),
    }


def time_steps(variant, microbatch, context, steps=10):
    import torch
    from model_100m.model import GPT, ModelConfig

    from kernels import fused

    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=32768, block_size=context)).cuda()
    if "fused" in variant:
        fused.fuse(model)
    if "compiled" in variant:
        model.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    tokens = torch.randint(4, 32768, (microbatch, context), device="cuda")
    labels = torch.randint(4, 32768, (microbatch, context), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, labels) / labels.numel()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / steps * 1000
    return {
        "step_ms": step_ms,
        "tokens_per_second": microbatch * context / step_ms * 1000,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@app.function(image=image, **limits)
def bench(deadline, microbatch=16, context=2048):
    import torch

    if time.time() > deadline:
        raise RuntimeError("Launch deadline passed; refusing work")
    report = {"device": torch.cuda.get_device_name(), "ops": op_checks(), "model": model_check()}
    report["steps"] = {}
    for variant in ("eager", "fused", "compiled", "fused+compiled"):
        try:
            report["steps"][variant] = time_steps(variant, microbatch, context)
        except Exception as exc:  # noqa: BLE001 -- record and continue with other variants
            report["steps"][variant] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        torch.cuda.empty_cache()
    return report


@app.local_entrypoint()
def main(output: str = "artifacts/kernels/bench.json"):
    report = bench.remote(time.time() + LAUNCH_LIMIT_SECONDS)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
