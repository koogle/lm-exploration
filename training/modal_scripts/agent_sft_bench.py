"""Throughput of one agent-SFT training step (`agent_sft_worker`) on one H100, by variant.

Run from training/:
    modal run modal_scripts/agent_sft_bench.py [--bases Qwen/Qwen3-0.6B-Base,...]

Each variant runs the worker's step on one synthetic trajectory per length (random
token ids, 30% target tokens): forward through the trunk in bf16 autocast over fp32
weights, the loss, backward. Variants:

- `worker`: as `agent_sft_worker` trains (gradient checkpointing, chunked
  cross-entropy recomputed in backward).
- `no-checkpointing`: the same without gradient checkpointing.
- `liger`: Liger's Triton kernels for RMSNorm, SwiGLU and RoPE, plus its fused linear
  cross-entropy (no [tokens, vocab] logits), with gradient checkpointing.
- `liger-no-checkpointing`: the same without gradient checkpointing.

Prints one JSON line per (base, variant, length): seconds per step (median of three
after a warmup), tokens per second and peak memory, or the error (for example out of
memory). The function has `retries=0` and a one-hour timeout; a supervisor cancels every
call six hours after launch, queued ones included.
"""

import json
import time

import modal
from model_hybrid import modal_benchmarks

from modal_scripts.modal_agent_sft import train_image as image
from modal_scripts.modal_agentic import REFERENCE_MODELS, hf_cache

app = modal.App("lm-exploration-agent-sft-bench")
app.include(modal_benchmarks.app)  # the deadline supervisor
DEADLINE_SECONDS = 6 * 60 * 60  # queued calls included
LENGTHS = (15000, 40960)  # the scale-500k mean trajectory, and the context
VARIANTS = ("worker", "no-checkpointing", "liger", "liger-no-checkpointing")
LOSS_CHUNK = 4096  # agent_sft_worker's loss_chunk


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=3600, retries=0,
    volumes={"/hf": hf_cache},
)  # fmt: skip
def bench(base, variant, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM

    liger = variant.startswith("liger")
    if liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=False,
        )  # fmt: skip
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        fused = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=REFERENCE_MODELS[base], dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    if not variant.endswith("no-checkpointing"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    head = model.get_output_embeddings()

    def chunk_loss(hidden, targets, weights):
        losses = torch.nn.functional.cross_entropy(head(hidden).float(), targets, reduction="none")
        return (losses * weights).sum()

    def step(ids, mask):
        inputs, targets, weights = ids[None, :-1], ids[1:], mask[1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=inputs).last_hidden_state[0]
            if liger:
                labels = torch.where(weights > 0, targets, -100)
                total = fused(head.weight, hidden, labels)
            else:
                total = hidden.new_zeros((), dtype=torch.float32)
                for begin in range(0, hidden.shape[0], LOSS_CHUNK):
                    end = begin + LOSS_CHUNK
                    total = total + checkpoint(
                        chunk_loss, hidden[begin:end], targets[begin:end], weights[begin:end],
                        use_reentrant=False,
                    )  # fmt: skip
        (total / weights.sum()).backward()
        model.zero_grad(set_to_none=True)

    results = []
    for length in LENGTHS:
        generator = torch.Generator(device="cuda").manual_seed(0)
        ids = torch.randint(0, 150000, (length,), device="cuda", generator=generator)
        mask = (torch.rand(length, device="cuda", generator=generator) < 0.3).float()
        record = {"base": base, "variant": variant, "length": length}
        try:
            torch.cuda.reset_peak_memory_stats()
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
        except Exception as error:  # noqa: BLE001
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(json.dumps(record), flush=True)
        results.append(record)
    return results


@app.local_entrypoint()
def main(bases: str = "Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base"):
    deadline = time.time() + DEADLINE_SECONDS
    calls = [bench.spawn(b, v, deadline) for b in bases.split(",") for v in VARIANTS]
    ids = [call.object_id for call in calls]
    supervisor = modal_benchmarks.enforce_deadline.spawn(ids, deadline)
    print(json.dumps({"calls": ids, "supervisor": supervisor.object_id, "deadline_unix": deadline}))
    for call in calls:
        for record in call.get():
            print(json.dumps(record))
