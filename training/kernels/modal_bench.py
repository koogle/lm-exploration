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
    # RMSNorm: FP32 residual in, BF16 out.
    x = torch.randn(4096, 768, device="cuda", requires_grad=True)
    w = (1 + 0.1 * torch.randn(768, device="cuda")).requires_grad_()
    dy = torch.randn(4096, 768, device="cuda").bfloat16()
    reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w
    gx, gw = torch.autograd.grad(reference, (x, w), dy.float())
    y = fused.RMSNormFunction.apply(x, w, 1e-5)
    fx, fw = torch.autograd.grad(y, (x, w), dy)
    results["rmsnorm"] = {
        "forward": relative_error(y, reference),
        "dx": relative_error(fx, gx),
        "dw": relative_error(fw, gw),
    }
    # QKV split + RoPE against the model's own rotary().
    qkv = torch.randn(2, 512, 3 * 768, device="cuda").bfloat16().requires_grad_()
    heads = 12

    def reference_qkv(t):
        q, k, v = t.float().chunk(3, -1)
        q, k, v = [z.view(2, 512, heads, 64).transpose(1, 2) for z in (q, k, v)]
        return rotary(q), rotary(k), v

    grads = [torch.randn(2, heads, 512, 64, device="cuda").bfloat16() for _ in range(3)]
    expected = reference_qkv(qkv)
    expected_grad = torch.autograd.grad(expected, qkv, [g.float() for g in grads])[0]
    actual = fused.QKVRopeFunction.apply(qkv, heads)
    actual_grad = torch.autograd.grad(actual, qkv, grads)[0]
    results["qkv_rope"] = {
        **{name: relative_error(a, e) for name, a, e in zip("qkv", actual, expected)},
        "dqkv": relative_error(actual_grad, expected_grad),
    }
    # SwiGLU activation.
    g = torch.randn(8192, 2048, device="cuda").bfloat16().requires_grad_()
    u = torch.randn(8192, 2048, device="cuda").bfloat16().requires_grad_()
    dh = torch.randn(8192, 2048, device="cuda").bfloat16()
    reference = F.silu(g.float()) * u.float()
    rg, ru = torch.autograd.grad(reference, (g, u), dh.float())
    h = fused.SwiGLUFunction.apply(g, u)
    ag, au = torch.autograd.grad(h, (g, u), dh)
    results["swiglu"] = {
        "forward": relative_error(h, reference),
        "dgate": relative_error(ag, rg),
        "dup": relative_error(au, ru),
    }
    # Cross-entropy with ignored targets.
    logits = (3 * torch.randn(4096, 32768, device="cuda")).bfloat16().requires_grad_()
    targets = torch.randint(4, 32768, (4096,), device="cuda")
    targets[::7] = -100
    reference = F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="sum")
    (rgrad,) = torch.autograd.grad(reference, logits)
    loss = fused.CrossEntropyFunction.apply(logits, targets)
    (agrad,) = torch.autograd.grad(loss, logits)
    results["cross_entropy"] = {
        "loss": relative_error(loss, reference),
        "dlogits": relative_error(agrad, rgrad),
    }
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
