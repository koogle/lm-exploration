"""One rank of `modal_agent_sft.train_mix`: data-parallel full fine-tune of a Qwen3 base.

Started by `torch.distributed.run` with a config path. Every rank holds the whole model
(fp32 weights, bf16 autocast, AdamW) with Liger's Triton kernels for RMSNorm, SwiGLU and
RoPE and its fused output-layer cross-entropy (no [tokens, vocab] logits), and reads the
mix's memory-mapped tokens. `balanced_steps` cuts the mix's training order
(`order.npy`) into steps of `world * accumulation` trajectories on average and gives
each rank about the same work per step (trajectories differ from 1k to 40k tokens and
every step waits for the slowest rank). Trajectories longer than
`checkpoint_above_tokens` run with gradient checkpointing (none when it is null). The
loss is summed cross-entropy on assistant tokens divided by the step's target tokens
over all ranks, and gradients are summed across ranks with all-reduce, so each step
averages over every target token in it.
Rank 0 writes metrics, the bf16 model and the summary. Every `checkpoint_steps` and at the
time budget it saves weights, optimizer state and the step to `checkpoint/state.pt`; a
later launch with the same output folder loads them and continues at that step.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def step_cost(length, quadratic_tokens):
    """Relative time of a trajectory: linear in tokens, plus attention's quadratic term
    (`quadratic_tokens` is the length at which the two are equal)."""
    return length + length * length / quadratic_tokens


def balanced_steps(lengths, world, per_step, window_steps, quadratic_tokens):
    """Steps of `per_step` trajectories on average, as `world` lists of positions in
    the training order, one per rank. Each window of `window_steps` steps assigns its
    trajectories, costliest first, to the least-loaded (step, rank) slot, so every rank
    of a step does about the same work; a trailing partial step is dropped."""
    import heapq

    usable = len(lengths) - len(lengths) % per_step
    steps = []
    for begin in range(0, usable, per_step * window_steps):
        positions = range(begin, min(begin + per_step * window_steps, usable))
        slots = [[] for _ in range(len(positions) // per_step * world)]
        heap = [(0.0, slot) for slot in range(len(slots))]
        costs = {p: step_cost(int(lengths[p]), quadratic_tokens) for p in positions}
        for position in sorted(positions, key=lambda p: (-costs[p], p)):
            load, slot = heapq.heappop(heap)
            slots[slot].append(position)
            heapq.heappush(heap, (load + costs[position], slot))
        steps += [slots[i : i + world] for i in range(0, len(slots), world)]
    return steps


def save_model(model, base, revision, context, folder):
    """The fine-tuned model in bf16 for vLLM, with the base's tokenizer."""
    from transformers import AutoTokenizer

    # Trained at the mix's context, which may exceed the base config's positions.
    model.config.max_position_embeddings = max(model.config.max_position_embeddings, context)
    model.config.use_cache = True
    # Turns end with <|im_end|>; the base generation config stops only at <|endoftext|>.
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    model.generation_config.eos_token_id = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.eos_token_id,
    ]
    model.to(torch.bfloat16).save_pretrained(folder)
    tokenizer.save_pretrained(folder)


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    settings = config["settings"]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
        apply_liger_kernel_to_qwen3,
    )
    from transformers import AutoModelForCausalLM

    mix = Path(config["mix"])
    offsets = np.load(mix / "offsets.npy")
    tokens = np.memmap(mix / "tokens.u32", dtype=np.uint32, mode="r")
    masks = np.memmap(mix / "mask.u8", dtype=np.uint8, mode="r")
    order = np.load(mix / "order.npy")
    schedule = balanced_steps(
        np.diff(offsets)[order],
        world,
        world * settings["accumulation"],
        settings["balance_window_steps"],
        settings["balance_quadratic_tokens"],
    )
    steps = len(schedule) * settings["epochs"]

    apply_liger_kernel_to_qwen3(
        rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
        fused_linear_cross_entropy=False,
    )  # fmt: skip
    fused_loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    model = AutoModelForCausalLM.from_pretrained(
        config["base"], revision=config["revision"], dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    checkpoint_above = settings["checkpoint_above_tokens"]
    model.config.use_cache = False
    model.train()
    for parameter in model.parameters():  # identical start on every rank
        dist.broadcast(parameter.data, 0)
    head = model.get_output_embeddings()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=settings["weight_decay"],
        fused=True,
    )

    def learning_rate(step):
        peak, floor = settings["learning_rate"], settings["min_learning_rate_fraction"]
        if step < settings["warmup_steps"]:
            return peak * (step + 1) / settings["warmup_steps"]
        progress = (step - settings["warmup_steps"]) / max(1, steps - settings["warmup_steps"])
        return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))

    def example(index):
        start, end = int(offsets[index]), int(offsets[index + 1])
        return np.asarray(tokens[start:end], dtype=np.int64), np.asarray(
            masks[start:end], dtype=np.float32
        )

    out = Path(config["out"])
    state_path = out / "checkpoint" / "state.pt"
    start, earlier_seconds, metrics = 0, 0.0, []
    if state_path.exists():
        state = torch.load(state_path, map_location="cuda", mmap=True, weights_only=True)
        if state["mix"] != config["mix"] or state["planned_steps"] != steps:
            raise ValueError(f"{state_path} belongs to another mix or schedule")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, earlier_seconds = state["step"], state["seconds"]
        del state
        with open(out / "metrics.jsonl") as handle:
            metrics = [json.loads(line) for line in handle][:start]
        if rank == 0:
            print(json.dumps({"resumed_at_step": start}), flush=True)

    def save_checkpoint(done):
        if rank == 0:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            partial = state_path.with_suffix(".partial")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": done,
                "planned_steps": steps,
                "mix": config["mix"],
                "seconds": earlier_seconds + time.time() - started,
            }
            torch.save(state, partial)
            partial.rename(state_path)
            with open(out / "metrics.jsonl", "w") as handle:
                handle.writelines(json.dumps(m) + "\n" for m in metrics)
        dist.barrier()

    started, stopped = time.time(), None
    for step in range(start, steps):
        stop = torch.tensor(int(time.time() >= config["stop_unix"]), device="cuda")
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            stopped = "time budget"
            save_checkpoint(step)
            break
        began = time.perf_counter()
        batch = [example(order[p]) for p in schedule[step % len(schedule)][rank]]
        loaded = time.perf_counter()
        local_targets = sum(float(mask[1:].sum()) for _, mask in batch)
        counts = torch.tensor(
            [local_targets, sum(len(ids) for ids, _ in batch)], device="cuda", dtype=torch.float64
        )
        dist.all_reduce(counts)
        global_targets = counts[0].item()
        loss_sum = 0.0
        for ids, mask in batch:
            # Lengths padded to a multiple of `pad_to_multiple` give the attention and
            # Triton kernels a few shapes to plan for instead of a new one per
            # trajectory; the padding comes last and is not a target, so under causal
            # attention it changes nothing before it.
            pad = -(len(ids) - 1) % settings["pad_to_multiple"]
            ids = np.concatenate([ids, np.zeros(pad, ids.dtype)])
            mask = np.concatenate([mask, np.zeros(pad, mask.dtype)])
            inputs = torch.from_numpy(ids[:-1]).cuda()[None]
            targets = torch.from_numpy(ids[1:]).cuda()
            labels = torch.where(torch.from_numpy(mask[1:]).cuda() > 0, targets, -100)
            recompute = checkpoint_above is not None and len(ids) > checkpoint_above
            if recompute != model.is_gradient_checkpointing:
                if recompute:
                    model.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_disable()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=inputs).last_hidden_state[0]
                total = fused_loss(head.weight, hidden, labels)
            (total / global_targets).backward()
            loss_sum += total.item()
        torch.cuda.synchronize()
        computed = time.perf_counter()
        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
        torch.cuda.synchronize()
        reduced = time.perf_counter()
        loss = torch.tensor(loss_sum, device="cuda", dtype=torch.float64)
        dist.all_reduce(loss)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        # Per-rank phase times: data, forward and backward, gradient all-reduce, optimizer.
        phases = torch.tensor(
            [loaded - began, computed - loaded, reduced - computed, time.perf_counter() - reduced],
            device="cuda",
        )
        gathered = [torch.zeros_like(phases) for _ in range(world)]
        dist.all_gather(gathered, phases)
        if rank == 0:
            gathered = torch.stack(gathered).tolist()
            record = {
                "step": step + 1,
                "loss": round(loss.item() / global_targets, 5),
                "grad_norm": round(float(norm), 4),
                "learning_rate": learning_rate(step),
                "tokens": int(counts[1].item()),
                "target_tokens": int(global_targets),
                "seconds": round(earlier_seconds + time.time() - started, 1),
                "phase_seconds": {
                    name: [
                        round(min(r[i] for r in gathered), 3),
                        round(max(r[i] for r in gathered), 3),
                    ]
                    for i, name in enumerate(("data", "compute", "all_reduce", "optimizer"))
                },
            }
            metrics.append(record)
            if step % 10 == 0:
                print(json.dumps(record), flush=True)
        if (step + 1) % settings["checkpoint_steps"] == 0 and step + 1 < steps:
            save_checkpoint(step + 1)
    if rank == 0:
        save_model(model, config["base"], config["revision"], settings["context"], out / "model")
        with open(out / "metrics.jsonl", "w") as handle:
            handle.writelines(json.dumps(m) + "\n" for m in metrics)
        manifest = json.loads((Path(config["mix"]) / "manifest.json").read_text())
        summary = {
            "base": config["base"],
            "revision": config["revision"],
            "mix": manifest["mix"],
            "mix_counts": manifest["counts"],
            "format": "mix: open-swe and terminus",
            "examples": manifest["examples"],
            "tokens": manifest["tokens"],
            "target_tokens": manifest["target_tokens"],
            "steps": metrics[-1]["step"] if metrics else 0,
            "planned_steps": steps,
            "stopped": stopped,
            "settings": settings,
            "world": world,
            "final_loss": metrics[-1]["loss"] if metrics else None,
            "seconds": round(earlier_seconds + time.time() - started),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
