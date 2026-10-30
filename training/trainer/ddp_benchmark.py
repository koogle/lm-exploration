"""One-host DDP throughput trial, launched with torchrun by the Modal worker."""

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from training_data.subword import EOS_TOKEN_ID, LOSS_IGNORE_INDEX


def trial(manifest, microbatch, timed_steps, output, benchmark_replay=False):
    import torch
    import torch.distributed as dist
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from training_data.token_stream import TokenMixture

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        # Unequal nonpadding counts exercise the required world/global-token scaling.
        torch.manual_seed(123)
        tiny = GPT(
            ModelConfig(
                vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, eos_id=EOS_TOKEN_ID
            )
        ).cuda()
        reference = copy.deepcopy(tiny) if rank == 0 else None
        tiny_ddp = DDP(tiny, device_ids=[rank])
        generator = torch.Generator().manual_seed(77)
        full_x = torch.randint(4, 64, (6, 16), generator=generator).cuda()
        full_y = torch.randint(4, 64, (6, 16), generator=generator).cuda()
        for i in range(6):
            full_y[i, 10 + i :] = LOSS_IGNORE_INDEX
        count = int((full_y != LOSS_IGNORE_INDEX).sum())
        x, y = full_x.chunk(world)[rank], full_y.chunk(world)[rank]
        loss = F.cross_entropy(
            tiny_ddp(x).flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
        )
        (loss * world / count).backward()
        if rank == 0:
            ref_loss = (
                F.cross_entropy(
                    reference(full_x).flatten(0, 1),
                    full_y.flatten(),
                    ignore_index=LOSS_IGNORE_INDEX,
                    reduction="sum",
                )
                / count
            )
            ref_loss.backward()
            for p, q in zip(tiny.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
            torch.optim.AdamW(tiny.parameters(), lr=3e-4, fused=True).step()
            torch.optim.AdamW(reference.parameters(), lr=3e-4, fused=True).step()
            for p, q in zip(tiny.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)
        dist.barrier()
        del tiny_ddp, tiny, reference, full_x, full_y, x, y, loss
        torch.cuda.empty_cache()

        torch.manual_seed(42)
        config = ModelConfig(vocab_size=32768, block_size=2048, eos_id=EOS_TOKEN_ID)
        global_windows = 48
        local_windows = global_windows // world
        if global_windows % world or local_windows % microbatch:
            raise ValueError("Microbatch must divide the per-rank effective batch")
        data = (
            TokenMixture(manifest, context=config.block_size, allow_unreviewed=True)
            if rank == 0
            else None
        )
        model = GPT(config).cuda()
        ddp = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        measurements = []
        initial_data = data.state_dict() if rank == 0 else None
        replays = 0
        trajectory = hashlib.sha256()
        torch.cuda.reset_peak_memory_stats()
        for step in range(timed_steps + 2):
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            # One loader constructs the identical global token-weighted batch used on one GPU.
            if rank == 0:
                try:
                    full_x, full_y = data.next_batch(global_windows, "cuda")
                except StopIteration:
                    if not benchmark_replay:
                        raise
                    data.load_state_dict(initial_data)
                    replays += 1
                    full_x, full_y = data.next_batch(global_windows, "cuda")
                trajectory.update(json.dumps(data.state_dict(), sort_keys=True).encode())
                xs, ys = list(full_x.chunk(world)), list(full_y.chunk(world))
                global_count = (full_y != LOSS_IGNORE_INDEX).sum()
            else:
                xs = ys = None
                global_count = torch.zeros((), dtype=torch.int64, device="cuda")
            x = torch.empty((local_windows, config.block_size), dtype=torch.int64, device="cuda")
            y = torch.empty_like(x)
            dist.scatter(x, xs, src=0)
            dist.scatter(y, ys, src=0)
            dist.broadcast(global_count, src=0)
            count = int(global_count)
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device="cuda")
            parts = list(zip(x.split(microbatch), y.split(microbatch)))
            for index, (mx, my) in enumerate(parts):
                sync = nullcontext() if index == len(parts) - 1 else ddp.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(
                            ddp(mx).flatten(0, 1),
                            my.flatten(),
                            ignore_index=LOSS_IGNORE_INDEX,
                            reduction="sum",
                        )
                    local_loss += loss.detach()
                    # DDP averages rank gradients; normalize by global loss tokens, not windows.
                    (loss * world / count).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True, foreach=True
            )
            optimizer.step()
            dist.all_reduce(local_loss)
            torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start
            if step >= 2 and rank == 0:
                measurements.append(
                    {
                        "seconds": elapsed,
                        "loss_tokens": count,
                        "loss": float(local_loss) / count,
                        "gradient_norm": float(norm),
                    }
                )
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device="cuda")
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            durations = [r["seconds"] for r in measurements]
            report = {
                "world_size": world,
                "microbatch_per_gpu": microbatch,
                "accumulation": local_windows // microbatch,
                "effective_windows": global_windows,
                "timed_steps": timed_steps,
                "warmup_steps": 2,
                "total_loss_tokens": sum(r["loss_tokens"] for r in measurements),
                "total_seconds": sum(durations),
                "median_step_seconds": statistics.median(durations),
                "min_step_seconds": min(durations),
                "max_step_seconds": max(durations),
                "peak_allocated_gib_per_gpu": float(peak),
                "measurements": measurements,
                "data_end": data.state_dict(),
                "benchmark_buffer_replays": replays,
                "data_trajectory_sha256": trajectory.hexdigest(),
                "masked_ddp_gradient_and_update_check": "passed",
                "device": torch.cuda.get_device_name(),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            report["loss_tokens_per_second"] = report["total_loss_tokens"] / report["total_seconds"]
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-replay", action="store_true")
    args = parser.parse_args()
    trial(args.manifest, args.microbatch, args.steps, args.output, args.benchmark_replay)
