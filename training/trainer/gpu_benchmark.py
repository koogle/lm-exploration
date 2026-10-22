"""Matched measured training throughput; no full pretraining is launched."""

import gc
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict

from training_data.subword import LOSS_IGNORE_INDEX


def explicit_causal_attention(query, key, value):
    """Masked-softmax reference for the SDPA kernel check."""
    import torch

    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    mask = torch.ones(query.size(-2), key.size(-2), device=query.device, dtype=torch.bool).tril()
    return scores.masked_fill(~mask, float("-inf")).softmax(dim=-1) @ value


def benchmark(
    manifest,
    requested_gpu,
    provenance,
    microbatches=(2, 4, 8, 16, 24, 48),
    sustained_steps=30,
    benchmark_replay=False,
):
    import torch
    from model_100m import model as modern
    from torch.nn import functional as F
    from training_data.token_stream import TokenMixture

    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Check the CUDA BF16 kernel against the explicit causal implementation.
    tensors = [
        torch.randn(2, 12, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    actual = modern.causal_attention(*tensors)
    expected = explicit_causal_attention(*tensors)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
    upstream = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, tensors, upstream),
        torch.autograd.grad(expected, tensors, upstream),
    ):
        torch.testing.assert_close(a, b, atol=0.04, rtol=0.08)
    del tensors, actual, expected, upstream
    gc.collect()
    torch.cuda.empty_cache()

    config = modern.ModelConfig(vocab_size=32768, block_size=2048)
    effective_windows = 48
    rows = []
    total_start = time.monotonic()

    def run(microbatch, timed_steps):
        torch.manual_seed(42)
        data = TokenMixture(manifest, context=config.block_size, allow_unreviewed=True)
        model = modern.GPT(config).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        initial_data = data.state_dict()
        replays = 0
        trajectory = hashlib.sha256()
        for step in range(timed_steps + 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            # Identical 48 windows per update on every device/microbatch size.
            try:
                full_x, full_y = data.next_batch(effective_windows, "cuda")
            except StopIteration:
                if not benchmark_replay:
                    raise
                data.load_state_dict(initial_data)
                replays += 1
                full_x, full_y = data.next_batch(effective_windows, "cuda")
            trajectory.update(json.dumps(data.state_dict(), sort_keys=True).encode())
            batches = list(zip(full_x.split(microbatch), full_y.split(microbatch)))
            count = sum(int((y != LOSS_IGNORE_INDEX).sum()) for _, y in batches)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for x, y in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(x).flatten(0, 1),
                        y.flatten(),
                        ignore_index=LOSS_IGNORE_INDEX,
                        reduction="sum",
                    )
                total_loss += float(loss.detach())
                (loss / count).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True, foreach=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= 2:
                measurements.append(
                    {
                        "seconds": elapsed,
                        "loss_tokens": count,
                        "loss": total_loss / count,
                        "gradient_norm": float(norm),
                    }
                )
        durations = [row["seconds"] for row in measurements]
        report = {
            "microbatch": microbatch,
            "accumulation": effective_windows // microbatch,
            "timed_steps": timed_steps,
            "warmup_steps": 2,
            "total_loss_tokens": sum(row["loss_tokens"] for row in measurements),
            "total_seconds": sum(durations),
            "median_step_seconds": statistics.median(durations),
            "min_step_seconds": min(durations),
            "max_step_seconds": max(durations),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "measurements": measurements,
            "data_start": initial_data,
            "data_end": data.state_dict(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "benchmark_buffer_replays": replays,
            "data_trajectory_sha256": trajectory.hexdigest(),
        }
        report["loss_tokens_per_second"] = report["total_loss_tokens"] / report["total_seconds"]
        return report

    for microbatch in microbatches:
        try:
            row = {"status": "ok", **run(microbatch, 6)}
        except torch.cuda.OutOfMemoryError:
            row = {"status": "out_of_memory", "microbatch": microbatch}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(
            json.dumps(
                {
                    "gpu": requested_gpu,
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("measurements", "data_start", "data_end")
                    },
                }
            ),
            flush=True,
        )
        if row["status"] == "out_of_memory":
            break
    successful = [r for r in rows if r["status"] == "ok"]
    if not successful:
        raise RuntimeError("No batch size fit")
    selected = max(successful, key=lambda row: row["loss_tokens_per_second"])
    gc.collect()
    torch.cuda.empty_cache()
    sustained = run(selected["microbatch"], sustained_steps)
    report = {
        "requested_gpu": requested_gpu,
        "device": torch.cuda.get_device_name(),
        "device_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "model_config": asdict(config),
        "effective_windows": effective_windows,
        "precision": "BF16 autocast / FP32 parameters",
        "attention": "PyTorch causal SDPA",
        "optimizer": "fused AdamW",
        "cpu": 4,
        "host_memory_mib": 24576,
        "cuda_attention_reference_check": "passed",
        "provenance": provenance,
        "sweep": rows,
        "sustained": sustained,
        "elapsed_seconds": time.monotonic() - total_start,
        "benchmark_replay": benchmark_replay,
        "notes": [
            "Optional bounded benchmark-buffer replay measures speed only; no weights are retained.",
            "Uses actual frozen-tokenizer pilot mixture batches, not synthetic tokens.",
            "Timings include loader, H2D, forward, backward, clipping and optimizer.",
            "Checkpoint/evaluation/preparation and cold startup are outside step timing.",
            "Two warmup steps excluded; fresh model and identical data prefix per trial.",
            "No full pretraining or production quality claim.",
        ],
    }
    print(
        json.dumps(
            {
                "gpu": requested_gpu,
                "sustained_loss_tokens_per_second": sustained["loss_tokens_per_second"],
            }
        ),
        flush=True,
    )
    return json.loads(json.dumps(report))
