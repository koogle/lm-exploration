"""One rank of `modal_agent_sft.train_mix`: data-parallel full fine-tune of a Qwen3 base.

Started by `torch.distributed.run` with a config path. Every rank holds the whole model
(fp32 weights, bf16 autocast, gradient checkpointing, AdamW) and reads the mix's
memory-mapped tokens; position i of the mix's training order (`order.npy`) goes to rank
i % world. A step is
`accumulation` trajectories per rank; the loss is summed cross-entropy on assistant
tokens divided by the step's target tokens over all ranks, and gradients are summed
across ranks with all-reduce, so each step averages over every target token in it.
The loss runs in chunks recomputed in backward, so the full logit matrix is never held.
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
from torch.utils.checkpoint import checkpoint


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
    from transformers import AutoModelForCausalLM

    mix = Path(config["mix"])
    offsets = np.load(mix / "offsets.npy")
    tokens = np.memmap(mix / "tokens.u32", dtype=np.uint32, mode="r")
    masks = np.memmap(mix / "mask.u8", dtype=np.uint8, mode="r")
    mine = np.load(mix / "order.npy")[rank::world]
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
    per_epoch = len(mine) // accumulation
    for step in range(start, steps):
        stop = torch.tensor(int(time.time() >= config["stop_unix"]), device="cuda")
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            stopped = "time budget"
            save_checkpoint(step)
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
                "seconds": round(earlier_seconds + time.time() - started, 1),
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
