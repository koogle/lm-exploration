"""Profile one 124M training step on an H100: where does GPU time go?

Run from training/:
    modal run -m kernels.modal_profile [--output artifacts/kernels/profile.json]
"""

import json
import time
from pathlib import Path

import modal

from kernels.image import LAUNCH_LIMIT_SECONDS, image, limits

app = modal.App("kernel-profile")


def step_fn(model, optimizer, tokens, labels):
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, labels) / labels.numel()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@app.function(image=image, **limits)
def profile(deadline, microbatch=16, context=2048, fused_kernels=False):
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.profiler import ProfilerActivity, profile

    if time.time() > deadline:
        raise RuntimeError("Launch deadline passed; refusing work")
    torch.manual_seed(0)
    results = {"device": torch.cuda.get_device_name(), "microbatch": microbatch, "context": context}
    for compiled in (False, True):
        model = GPT(ModelConfig(vocab_size=32768, block_size=context)).cuda()
        if fused_kernels:
            from kernels import fused

            fused.fuse(model)
        if compiled:
            model.compile()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        tokens = torch.randint(4, 32768, (microbatch, context), device="cuda")
        labels = torch.randint(4, 32768, (microbatch, context), device="cuda")
        for _ in range(3):
            step_fn(model, optimizer, tokens, labels)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            step_fn(model, optimizer, tokens, labels)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - start) / 10 * 1000
        name = "compiled" if compiled else "eager"
        entry = {
            "step_ms": step_ms,
            "tokens_per_second": microbatch * context / step_ms * 1000,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        if not compiled:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    step_fn(model, optimizer, tokens, labels)
                torch.cuda.synchronize()
            events = sorted(
                prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
            )
            total = sum(e.self_device_time_total for e in events)
            entry["top_kernels"] = [
                {
                    "name": e.key[:160],
                    "share": e.self_device_time_total / total,
                    "ms_per_step": e.self_device_time_total / 3 / 1000,
                    "calls_per_step": e.count / 3,
                }
                for e in events[:30]
            ]
        results[name] = entry
        del model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return results


@app.local_entrypoint()
def main(output: str = "artifacts/kernels/profile.json", fused_kernels: bool = False):
    report = profile.remote(time.time() + LAUNCH_LIMIT_SECONDS, fused_kernels=fused_kernels)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("eager",)}, indent=2))
    for kernel in report["eager"]["top_kernels"][:20]:
        print(f"{kernel['share']:6.1%} {kernel['ms_per_step']:7.2f} ms  {kernel['name'][:110]}")
