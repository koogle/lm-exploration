"""Throughput of one agent-SFT training step (`agent_sft_worker`) on one H100, by variant.

Run from training/:
    modal run modal_scripts/agent_sft_bench.py [--bases Qwen/Qwen3-0.6B-Base,...]

Each variant runs the worker's step on one synthetic trajectory per length (random
token ids, 30% target tokens; lengths are multiples of the worker's 1,024-token
padding): forward through the trunk in bf16 autocast over fp32 weights with Liger's
RMSNorm, SwiGLU and RoPE kernels, Liger's fused linear cross-entropy, backward, no
gradient checkpointing. Variants differ in attention and compilation:

- `worker`: as the worker trains (SDPA, PyTorch's choice of backend).
- `sdpa-flash`, `sdpa-cudnn`: SDPA restricted to FlashAttention-2 or cuDNN attention.
- `fa3`: FlashAttention-3 (Hopper) from the Hugging Face kernel hub
  (`kernels-community/flash-attn3`, pinned), causal over the whole sequence, which is
  exact for the worker's one end-padded trajectory per forward.
- `compiled-layers`: `worker` with each decoder layer compiled (`torch.compile`).

Prints one JSON line per (base, variant, length): seconds per step (median of three
after two warmups), tokens per second and peak memory, or the error. `profile` adds,
for `worker` at the first length, the CUDA time per kernel category and the top
kernels from `torch.profiler`. The function has `retries=0` and a one-hour timeout; a
supervisor cancels every call six hours after launch, queued ones included.
"""

import json
import time

import modal
from model_hybrid import modal_benchmarks

from modal_scripts.modal_agentic import REFERENCE_MODELS, hf_cache, reference_base
from workflows.images import add_project

app = modal.App("lm-exploration-agent-sft-bench")
app.include(modal_benchmarks.app)  # the deadline supervisor
image = add_project(reference_base.pip_install("liger-kernel==0.8.4", "kernels==0.17.2"))
DEADLINE_SECONDS = 6 * 60 * 60  # queued calls included
LENGTHS = (16384, 40960)  # about the scale-500k mean trajectory, and the context
VARIANTS = ("worker", "sdpa-flash", "sdpa-cudnn", "fa3", "compiled-layers")
FA3 = {
    "repo": "kernels-community/flash-attn3",
    "revision": "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0",
}
CATEGORIES = {  # kernel-name fragments, checked in order
    "attention": ("flash", "fmha", "attention", "cudnn"),
    "matmul": ("gemm", "cutlass", "xmma", "nvjet", "matmul"),
    "liger_triton": ("liger", "_kernel", "triton"),
}


def _fa3_attention():
    """An `AttentionInterface` function running FlashAttention-3, causal."""
    import importlib
    import sys

    import torch

    # The project's own `kernels` package (training/kernels) shadows the kernel hub's.
    project = [p for p in sys.path if p in ("", "/project")]
    sys.path[:] = [p for p in sys.path if p not in project]
    shadowed = sys.modules.pop("kernels", None)
    try:
        hub = importlib.import_module("kernels")
    finally:
        sys.path[:0] = project
        if shadowed is not None:
            sys.modules["kernels"] = shadowed
    flash = hub.get_kernel(FA3["repo"], revision=FA3["revision"])

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_):
        # [batch, heads, tokens, dim] -> [batch, tokens, heads, dim]
        q, k, v = (t.transpose(1, 2).to(torch.bfloat16) for t in (query, key, value))
        out = flash.flash_attn_func(q, k, v, softmax_scale=scaling, causal=True)
        out = out[0] if isinstance(out, tuple) else out
        return out, None

    return attention


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=3600, retries=0,
    volumes={"/hf": hf_cache},
)  # fmt: skip
def bench(base, variant, profile, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import contextlib

    import torch
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
        apply_liger_kernel_to_qwen3,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface, AutoModelForCausalLM

    apply_liger_kernel_to_qwen3(
        rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
        fused_linear_cross_entropy=False,
    )  # fmt: skip
    fused = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    implementation = "sdpa"
    if variant == "fa3":
        AttentionInterface.register("fa3", _fa3_attention())
        implementation = "fa3"
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=REFERENCE_MODELS[base], dtype=torch.float32,
        attn_implementation=implementation,
    ).cuda()  # fmt: skip
    model.config.use_cache = False
    model.train()
    if variant == "compiled-layers":
        for layer in model.model.layers:
            layer.compile()
    head = model.get_output_embeddings()
    backend = {
        "sdpa-flash": [SDPBackend.FLASH_ATTENTION],
        "sdpa-cudnn": [SDPBackend.CUDNN_ATTENTION],
    }.get(variant)

    def step(ids, mask):
        inputs, targets, weights = ids[None, :-1], ids[1:], mask[1:]
        labels = torch.where(weights > 0, targets, -100)
        restrict = sdpa_kernel(backend) if backend else contextlib.nullcontext()
        with restrict, torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=inputs).last_hidden_state[0]
            total = fused(head.weight, hidden, labels)
        (total / weights.sum()).backward()
        model.zero_grad(set_to_none=True)

    results = []
    for index, length in enumerate(LENGTHS):
        generator = torch.Generator(device="cuda").manual_seed(0)
        ids = torch.randint(0, 150000, (length + 1,), device="cuda", generator=generator)
        mask = (torch.rand(length + 1, device="cuda", generator=generator) < 0.3).float()
        record = {"base": base, "variant": variant, "length": length}
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                step(ids, mask)
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                step(ids, mask)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
            seconds = sorted(times)[1]
            record |= {
                "seconds": round(seconds, 3),
                "tokens_per_second": round(length / seconds),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            if profile and index == 0:
                record["profile"] = _profile(step, ids, mask)
        except Exception as error:  # noqa: BLE001
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(json.dumps(record), flush=True)
        results.append(record)
    return results


def _profile(step, ids, mask):
    """CUDA milliseconds of one step per kernel category, and the ten costliest kernels."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        step(ids, mask)
    kernels = {}
    for event in trace.key_averages():
        if event.device_time_total > 0:
            kernels[event.key] = kernels.get(event.key, 0.0) + event.device_time_total / 1000
    categories = dict.fromkeys([*CATEGORIES, "other"], 0.0)
    for name, milliseconds in kernels.items():
        lowered = name.lower()
        category = next(
            (c for c, parts in CATEGORIES.items() if any(p in lowered for p in parts)), "other"
        )
        categories[category] += milliseconds
    top = sorted(kernels.items(), key=lambda item: -item[1])[:10]
    return {
        "milliseconds_by_category": {k: round(v, 1) for k, v in categories.items()},
        "top_kernels": [[name[:120], round(ms, 1)] for name, ms in top],
    }


@app.local_entrypoint()
def main(bases: str = "Qwen/Qwen3-0.6B-Base", variants: str = ",".join(VARIANTS)):
    deadline = time.time() + DEADLINE_SECONDS
    calls = [
        bench.spawn(b, v, v == "worker", deadline)
        for b in bases.split(",")
        for v in variants.split(",")
    ]
    ids = [call.object_id for call in calls]
    supervisor = modal_benchmarks.enforce_deadline.spawn(ids, deadline)
    print(json.dumps({"calls": ids, "supervisor": supervisor.object_id, "deadline_unix": deadline}))
    for call in calls:
        for record in call.get():
            print(json.dumps(record))
