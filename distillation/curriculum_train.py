"""One-pass qualification of fresh students on newly generated accepted records."""

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from distillation.curriculum import training_text, verify


def make_windows(records, tokenizer, context=512):
    windows = []
    for record in records:
        ids = [1] + tokenizer.encode(training_text(record)).ids + [2]
        for start in range(0, len(ids) - 1, context):
            segment = ids[start : start + context + 1]
            windows.append((segment[:-1], segment[1:]))
    return windows


def run(corpus, output, commit):
    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import TOKENIZER_SHA256, load_frozen

    corpus = Path(corpus)
    output = Path(output)
    output.mkdir(exist_ok=False)
    raw = (corpus / "accepted.jsonl").read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    tasks = {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in (corpus / "tasks.jsonl").read_text().splitlines()
        )
    }
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if len(train) < 32 or len(validation) < 8:
        raise ValueError("Insufficient verified data for a qualification comparison")
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("Split leakage")
    tokenizer = load_frozen()
    random.Random(42).shuffle(train)
    windows = make_windows(train, tokenizer)
    val_windows = make_windows(validation, tokenizer)
    target_tokens = sum(len(y) for _, y in windows)
    settings = {
        "corpus": str(corpus),
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "seed": 42,
        "context": 512,
        "microbatch": 8,
        "accumulation": 2,
        "lr": 0.0003,
        "objective": "all-token sequence distillation CE, including prompts",
        "initialization": "random, not a continuation of pretrained weights",
        "epochs": 1,
        "unique_train_tokens": target_tokens,
        "train_records": len(train),
        "validation_records": len(validation),
        "scope": "pipeline qualification; insufficient data to estimate a scaling law",
    }
    (output / "request.json").write_text(json.dumps(settings, indent=2))
    shutil.copytree(
        "/project/distillation",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__", "results"),
    )
    for name in ("model_10m", "model_100m"):
        shutil.copytree(
            Path("/project") / name,
            output / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    commit()
    reports = []

    def train_one(layers, heads, width):
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        config = ModelConfig(
            vocab_size=32768, n_layer=layers, n_head=heads, n_embd=width, block_size=512
        )
        model = GPT(config).cuda()
        n = sum(p.numel() for p in model.parameters())
        folder = output / f"student-{n}"
        folder.mkdir()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["lr"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
        tokens = step = 0
        started = time.monotonic()
        evaluations = []

        def tensors(batch):
            length = max(len(y) for _, y in batch)
            x = torch.zeros((len(batch), length), dtype=torch.long, device="cuda")
            y = torch.full_like(x, -100)
            for i, (a, b) in enumerate(batch):
                x[i, : len(a)] = torch.tensor(a, device="cuda")
                y[i, : len(b)] = torch.tensor(b, device="cuda")
            return x, y

        def evaluate_and_save():
            model.eval()
            loss = count = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(val_windows), 8):
                    x, y = tensors(val_windows[start : start + 8])
                    loss += F.cross_entropy(
                        model(x).float().flatten(0, 1),
                        y.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    ).item()
                    count += (y != -100).sum().item()
            evaluation = {
                "tokens": tokens,
                "step": step,
                "validation_ce": loss / count,
                "validation_tokens": count,
            }
            evaluations.append(evaluation)
            generations = []
            seen = set()
            # Fixed one-per-family held-out prompts, before/mid/after training.
            for record in tasks.values():
                if record["split"] != "validation":
                    continue
                if record["family"] in seen:
                    continue
                seen.add(record["family"])
                prompt = f"### Problem\n{record['prompt']}\n### Python\n"
                ids = [1] + tokenizer.encode(prompt).ids
                generated = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(160):
                        token = (
                            model(torch.tensor([ids[-512:]], device="cuda"))[0, -1]
                            .argmax()
                            .item()
                        )
                        if token == 2:
                            break
                        ids.append(token)
                        generated.append(token)
                text = tokenizer.decode(generated, skip_special_tokens=False)
                code = text.split("### Explanation", 1)[0].strip()
                result = verify(code, tasks[record["id"]]["cases"])
                generations.append(
                    {
                        "id": record["id"],
                        "family": record["family"],
                        "prompt": prompt,
                        "continuation": text,
                        "token_ids": generated,
                        "verification": result,
                    }
                )
            evaluation["diagnostic_passes"] = sum(
                g["verification"]["passed"] for g in generations
            )
            evaluation["diagnostic_tasks"] = len(generations)
            with (folder / "validation.jsonl").open("a") as f:
                f.write(json.dumps(evaluation) + "\n")
            (folder / f"generations-{step:06d}.json").write_text(
                json.dumps(generations, indent=2)
            )
            torch.save(
                {
                    "format": "curriculum-student-v1",
                    "model_config": asdict(config),
                    "settings": settings,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "tokens": tokens,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "evaluation": evaluation,
                },
                folder / f"step-{step:09d}.pt",
            )
            print(json.dumps({"parameters": n, **evaluation}), flush=True)
            commit()
            model.train()

        evaluate_and_save()
        midpoint_saved = False
        for start in range(0, len(windows), 16):
            batch = windows[start : start + 16]
            batch_tokens = sum(len(y) for _, y in batch)
            optimizer.zero_grad(set_to_none=True)
            progress = (tokens + batch_tokens) / target_tokens
            warmup = min(1.0, progress / 0.1)
            decay = (
                0.1
                + 0.9 * (1 + math.cos(math.pi * max(0.0, (progress - 0.1) / 0.9))) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"] * warmup * decay
            train_loss = 0
            for sub in range(0, len(batch), 8):
                x, y = tensors(batch[sub : sub + 8])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = (
                        F.cross_entropy(
                            model(x).float().flatten(0, 1),
                            y.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        / batch_tokens
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            tokens += batch_tokens
            step += 1
            metric = {
                "step": step,
                "tokens": tokens,
                "train_ce": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            with (folder / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
            if tokens >= target_tokens / 2 and not midpoint_saved:
                evaluate_and_save()
                midpoint_saved = True
        evaluate_and_save()
        report = {
            "parameters": n,
            "tokens": tokens,
            "steps": step,
            "elapsed_seconds": time.monotonic() - started,
            "evaluations": evaluations,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "configuration": asdict(config),
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        commit()
        return report

    for layers, heads, width in ((8, 6, 384), (10, 8, 512)):
        reports.append(train_one(layers, heads, width))
        torch.cuda.empty_cache()
    result = {"status": "completed", "settings": settings, "students": reports}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    commit()
    return result
