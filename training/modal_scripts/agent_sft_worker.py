"""One rank of `modal_agent_sft.train_mix`: data-parallel full fine-tune of a Qwen3 base.

Started by `torch.distributed.run` with a config path. Every rank holds the whole model
(fp32 weights, bf16 autocast, gradient checkpointing, AdamW) and reads the mix's
memory-mapped tokens; example i of the shuffled mix goes to rank i % world. A step is
`accumulation` trajectories per rank; the loss is summed cross-entropy on assistant
tokens divided by the step's target tokens over all ranks, and gradients are summed
across ranks with all-reduce, so each step averages over every target token in it.
The loss runs in chunks recomputed in backward, so the full logit matrix is never held.
Rank 0 writes metrics, the bf16 model and the summary.
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
from torch.utils.checkpoint import checkpoint


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    settings = config["settings"]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mix = Path(config["mix"])
    offsets = np.load(mix / "offsets.npy")
    tokens = np.memmap(mix / "tokens.u32", dtype=np.uint32, mode="r")
    masks = np.memmap(mix / "mask.u8", dtype=np.uint8, mode="r")
    mine = list(range(rank, len(offsets) - 1, world))
    accumulation = settings["accumulation"]
    steps = len(mine) // accumulation
    steps_tensor = torch.tensor(steps, device="cuda")
    dist.all_reduce(steps_tensor, op=dist.ReduceOp.MIN)
    steps = int(steps_tensor) * settings["epochs"]

    model = AutoModelForCausalLM.from_pretrained(
        config["base"], revision=config["revision"], dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable()
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

    def chunk_loss(hidden, targets, weights):
        losses = torch.nn.functional.cross_entropy(head(hidden).float(), targets, reduction="none")
        return (losses * weights).sum()

    def example(index):
        start, end = int(offsets[index]), int(offsets[index + 1])
        return np.asarray(tokens[start:end], dtype=np.int64), np.asarray(
            masks[start:end], dtype=np.float32
        )

    started, metrics, stopped = time.time(), [], None
    per_epoch = len(mine) // accumulation
    for step in range(steps):
        stop = torch.tensor(int(time.time() >= config["stop_unix"]), device="cuda")
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            stopped = "time budget"
            break
        position = (step % per_epoch) * accumulation
        batch = [example(i) for i in mine[position : position + accumulation]]
        local_targets = sum(float(mask[1:].sum()) for _, mask in batch)
        counts = torch.tensor(
            [local_targets, sum(len(ids) for ids, _ in batch)], device="cuda", dtype=torch.float64
        )
        dist.all_reduce(counts)
        global_targets = counts[0].item()
        loss_sum = 0.0
        for ids, mask in batch:
            inputs = torch.from_numpy(ids[:-1]).cuda()[None]
            targets = torch.from_numpy(ids[1:]).cuda()
            weights = torch.from_numpy(mask[1:]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=inputs).last_hidden_state[0]
                total = hidden.new_zeros((), dtype=torch.float32)
                for begin in range(0, hidden.shape[0], settings["loss_chunk"]):
                    end = begin + settings["loss_chunk"]
                    if weights[begin:end].any():
                        total = total + checkpoint(
                            chunk_loss,
                            hidden[begin:end],
                            targets[begin:end],
                            weights[begin:end],
                            use_reentrant=False,
                        )
            (total / global_targets).backward()
            loss_sum += total.item()
        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
        loss = torch.tensor(loss_sum, device="cuda", dtype=torch.float64)
        dist.all_reduce(loss)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            record = {
                "step": step + 1,
                "loss": round(loss.item() / global_targets, 5),
                "grad_norm": round(float(norm), 4),
                "learning_rate": learning_rate(step),
                "tokens": int(counts[1].item()),
                "target_tokens": int(global_targets),
                "seconds": round(time.time() - started, 1),
            }
            metrics.append(record)
            if step % 10 == 0:
                print(json.dumps(record), flush=True)
    if rank == 0:
        out = Path(config["out"])
        model.to(torch.bfloat16).save_pretrained(out / "model")
        AutoTokenizer.from_pretrained(config["base"], revision=config["revision"]).save_pretrained(
            out / "model"
        )
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
            "steps": len(metrics),
            "planned_steps": steps,
            "stopped": stopped,
            "settings": settings,
            "world": world,
            "final_loss": metrics[-1]["loss"] if metrics else None,
            "seconds": round(time.time() - started),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
