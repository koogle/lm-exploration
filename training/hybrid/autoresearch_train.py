"""Matched proxy training with resumable data, fixed schedule, and bounded runtime."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .autoresearch_model import build, state_bytes
from .autoresearch_specs import PROXY, FINAL_STEPS, SCREEN_STEPS, TOKENS_PER_STEP
from .prepared_data import PreparedMixture, heldout_index


def train(
    run,
    variant,
    seed,
    deadline,
    recipe,
    volume,
    stop_step=SCREEN_STEPS,
    source=None,
    config=None,
    stage="screen",
    total_steps=FINAL_STEPS,
):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = dict(PROXY if config is None else config)
    # A fixed 4K curriculum for screening and confirmation. Longer-context
    # adaptation is an explicit separate experiment rather than silent drift.
    context = cfg["block_size"]
    batch = TOKENS_PER_STEP // context
    assert batch >= 1 and TOKENS_PER_STEP % context == 0
    recipe = dict(recipe, context=context)
    stream = PreparedMixture(recipe, "/data/" + recipe["name"], wait_seconds=120)
    validation = PreparedMixture(recipe, "/data/" + recipe["name"], wait_seconds=120)
    validation.orders = {n: [heldout_index(recipe["sources"][n])] for n in validation.names}
    batches = [validation.next_batch(batch) for _ in range(24)]
    folder = Path("/runs") / run / stage / f"{variant}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model = build(cfg, variant, seed).cuda()
    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    start = 0
    if source:
        ck = torch.load(source, map_location="cpu", weights_only=True)
        assert ck["variant"] == variant and ck["seed"] == seed and ck["config"] == cfg
        assert ck["total_steps"] == total_steps
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        stream.load_state_dict(ck["data_state"])
        start = ck["step"]
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        del ck

    def evaluate():
        sums = {n: [0.0, 0] for n in validation.names}
        doc_ce = []
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for xx, yy, names in batches:
                # Source-wise rows permit paired diagnostics instead of only a global scalar.
                for x, y, name in zip(xx, yy, names):
                    out = model(
                        torch.as_tensor(x[None], device="cuda"),
                        torch.as_tensor(y[None], device="cuda"),
                    )
                    ce, count = out["ce"].item(), out["count"].item()
                    sums[name][0] += ce
                    sums[name][1] += count
                    doc_ce.append(ce / count)
        model.train()
        return {
            "ce": sum(v[0] for v in sums.values()) / sum(v[1] for v in sums.values()),
            "sources": {n: v[0] / v[1] for n, v in sums.items() if v[1]},
            "window_ce": doc_ce,
        }

    result = {
        "variant": variant,
        "seed": seed,
        "stage": stage,
        "config": cfg,
        "effective_config": asdict(model.config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "nonembedding_parameters": sum(p.numel() for p in model.parameters())
        - cfg["vocab_size"] * cfg["n_embd"],
        "state_bytes": state_bytes(model),
        "planned_steps": stop_step,
        "schedule_steps": total_steps,
        "initial": evaluate(),
        "status": "running",
        "curve": [],
        "steps_completed": start,
    }
    started = time.time()
    optimizer_seconds = 0.0
    timed_tokens = 0
    loss_sum = 0.0
    count_sum = 0

    def save(step, checkpoint=False):
        result["steps_completed"] = step
        result["wall_seconds"] = time.time() - started
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        (folder / "metrics.json").write_text(json.dumps(result))
        if checkpoint:
            tmp = folder / "checkpoint.tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "variant": variant,
                    "seed": seed,
                    "config": cfg,
                    "step": step,
                    "total_steps": total_steps,
                    "data_state": stream.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                tmp,
            )
            tmp.replace(folder / "checkpoint.pt")
        volume.commit()

    save(start)
    for step in range(start, stop_step):
        if time.time() > deadline - 240:
            result["status"] = "deadline"
            save(step, True)
            return result
        lr = (
            3e-4
            * min(1.0, (step + 1) / 400)
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
        )
        for group in opt.param_groups:
            group["lr"] = lr
        xx, yy, _ = stream.next_batch(batch)
        x, y = torch.as_tensor(xx, device="cuda"), torch.as_tensor(yy, device="cuda")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
            loss = out["loss"] / out["count"]
        if not torch.isfinite(loss):
            result["status"] = "nonfinite"
            save(step, True)
            return result
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        if step >= start + 10:
            optimizer_seconds += time.perf_counter() - tick
            timed_tokens += TOKENS_PER_STEP
        loss_sum += out["ce"].item()
        count_sum += out["count"].item()
        if step == start or (step + 1) % 1024 == 0 or step + 1 == stop_step:
            row = {
                "step": step + 1,
                "tokens": (step + 1) * TOKENS_PER_STEP,
                "validation": evaluate(),
                "train_ce": loss_sum / count_sum,
                "grad_norm": norm.item(),
                "lr": lr,
                "tokens_per_second": timed_tokens / optimizer_seconds
                if optimizer_seconds
                else None,
                "wall_seconds": time.time() - started,
            }
            result["curve"].append(row)
            result["status"] = "complete" if step + 1 == stop_step else "running"
            print(
                variant,
                seed,
                stage,
                {k: v for k, v in row.items() if k != "validation"},
                "val",
                row["validation"]["ce"],
                flush=True,
            )
            save(step + 1, step == start or (step + 1) % 4096 == 0 or step + 1 == stop_step)
            loss_sum = count_sum = 0.0
    return result
