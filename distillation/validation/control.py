"""Fine-tune the pretrained Qwen3.5-0.8B-Base control on the same packed stream.

Full-parameter training with the students' chunked loss (CE only by default), BF16
autocast over FP32 weights. Saves a text-only checkpoint that vLLM can serve.

Usage: python control.py SETTINGS.json MANIFEST.json OUT_DIR
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import CONTROL, CONTROL_REVISION, QWEN_ENDOFTEXT, QWEN_VOCAB
from student import (
    Stream,
    Vocabulary,
    batches,
    chunk_losses,
    learning_rate,
    to_device,
    validate,
)
from torch.utils.checkpoint import checkpoint


def losses(model, inputs, labels, top_ids, top_lp, chunk=1024):
    hidden = model.model(input_ids=inputs).last_hidden_state.flatten(0, 1)
    parts = zip(
        hidden.split(chunk),
        labels.flatten().split(chunk),
        top_ids.flatten(0, 1).split(chunk),
        top_lp.flatten(0, 1).split(chunk),
    )
    return sum(checkpoint(chunk_losses, model, *p, use_reentrant=False) for p in parts)


def main(settings_path, manifest_path, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = json.loads(Path(settings_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    out_dir = Path(out_dir)
    log_path = out_dir / "metrics.jsonl"
    torch.manual_seed(settings["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        CONTROL,
        revision=CONTROL_REVISION,
        dtype=torch.float32,
        cache_dir="/runs/hf-cache",
    ).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        CONTROL, revision=CONTROL_REVISION, cache_dir="/runs/hf-cache"
    )
    vocab = Vocabulary.full(QWEN_VOCAB, QWEN_ENDOFTEXT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    per_step = settings["batch_windows"] * settings["window"]
    steps = settings["positions"] // per_step
    accumulation = settings["batch_windows"] // settings["micro_windows"]
    warmup = max(20, steps // 50)
    weight = settings["kd_weight"]
    byte_lengths = np.load(manifest["byte_lengths"])
    before = validate(
        model,
        manifest["heldout"],
        vocab,
        byte_lengths,
        settings["validation_positions"],
        loss_fn=losses,
    )
    stream = batches(
        Stream(manifest["train"], vocab, settings["window"]), settings["micro_windows"]
    )
    started = time.time()
    labels_seen = 0
    model.train()
    with log_path.open("a") as log:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, steps, settings["lr"], warmup)
            micro = [to_device(next(stream), "cuda") for _ in range(accumulation)]
            count = sum(int((b[1] != -100).sum()) for b in micro)
            sums = torch.zeros(3, device="cuda")
            for batch in micro:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    part = losses(model, *batch)
                (((1 - weight) * part[0] + weight * part[1]) / count).backward()
                sums += part.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            labels_seen += count
            if step % settings["log_every"] == 0 or step == steps - 1:
                ce, soft, entropy = (sums / count).tolist()
                elapsed = time.time() - started
                entry = {
                    "event": "step",
                    "step": step,
                    "steps": steps,
                    "ce": ce,
                    "kl": soft - entropy,
                    "grad_norm": float(norm),
                    "labels": labels_seen,
                    "positions_per_second": (step + 1) * per_step / elapsed,
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
    metrics = validate(
        model,
        manifest["heldout"],
        vocab,
        byte_lengths,
        settings["validation_positions"],
        loss_fn=losses,
    )
    model.config.use_cache = True
    model.save_pretrained(out_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(out_dir / "model")
    summary = {
        "steps": steps,
        "positions": steps * per_step,
        "labels": labels_seen,
        "train_seconds": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_before": before,
        "validation": metrics,
    }
    (out_dir / "train-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
