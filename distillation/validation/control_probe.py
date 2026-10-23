"""Measure Qwen3.5-0.8B fine-tuning throughput and which fast kernels are active.

Usage: python control_probe.py OUT.json
"""

import json
import sys
import time

import torch
from common import CONTROL, CONTROL_REVISION


def main(out):
    from transformers import AutoModelForCausalLM
    from transformers.utils import import_utils

    report = {
        name: bool(getattr(import_utils, name)())
        for name in (
            "is_flash_linear_attention_available",
            "is_causal_conv1d_available",
        )
        if hasattr(import_utils, name)
    }
    from importlib import metadata

    for package in ("flash-linear-attention", "causal-conv1d", "torch", "transformers"):
        try:
            report[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report[package] = None
    import importlib
    import traceback

    for module, name in (
        ("fla", "chunk_gated_delta_rule"),
        ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ):
        try:
            report[f"import {module}.{name}"] = callable(
                getattr(importlib.import_module(module), name, None)
            )
        except Exception:  # noqa: BLE001 - the probe reports why the fast path is off
            report[f"import {module}.{name}"] = traceback.format_exc()[-1500:]
    print(json.dumps(report, indent=1), flush=True)
    if "--imports-only" in sys.argv:
        return
    from control import losses

    model = AutoModelForCausalLM.from_pretrained(
        CONTROL,
        revision=CONTROL_REVISION,
        dtype=torch.float32,
        cache_dir="/runs/hf-cache",
    ).cuda()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
    vocab = model.config.vocab_size
    for checkpointing in (True, False):
        if checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        for micro in (4, 8):
            torch.cuda.reset_peak_memory_stats()
            shape = (micro, 2048)
            inputs = torch.randint(0, vocab, shape, device="cuda")
            labels = torch.randint(0, vocab, shape, device="cuda")
            ids = torch.randint(0, vocab, (*shape, 16), device="cuda")
            lps = torch.full((*shape, 16), -5.0, device="cuda")
            try:
                for step in range(5):
                    if step == 2:
                        torch.cuda.synchronize()
                        started = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        part = losses(model, inputs, labels, ids, lps)
                    (part[0] / labels.numel()).backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                rate = 3 * micro * 2048 / (time.time() - started)
                report[f"ckpt={checkpointing},micro={micro}"] = {
                    "tokens_per_second": rate,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            except torch.OutOfMemoryError:
                report[f"ckpt={checkpointing},micro={micro}"] = "out of memory"
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            print(json.dumps(report), flush=True)
    # One profiled step at the measured configuration.
    model.gradient_checkpointing_enable()
    inputs = torch.randint(0, vocab, (4, 2048), device="cuda")
    labels = torch.randint(0, vocab, (4, 2048), device="cuda")
    ids = torch.randint(0, vocab, (4, 2048, 16), device="cuda")
    lps = torch.full((4, 2048, 16), -5.0, device="cuda")
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profiler:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            part = losses(model, inputs, labels, ids, lps)
        (part[0] / labels.numel()).backward()
        torch.cuda.synchronize()
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(table, flush=True)
    report["profile_top"] = table
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
