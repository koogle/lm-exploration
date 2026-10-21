"""Resumable exact-token student segments over incrementally published shards."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from distillation.scale_data import atomic_json, available_chunks, read_chunk

LARGE_PARENT = "qwen4b-shared-50m-20261021-01/step-000007162.pt"


class Stream:
    def __init__(self, root, cursor=None, large=False):
        self.root, self.large = root, large
        self.chunks = available_chunks(root)
        self.cursor = dict(cursor or {"chunk": 0, "record": 0, "offset": 0})
        self.cache_index, self.cache = -1, None

    def next(self, limit=512):
        while self.cursor["chunk"] < len(self.chunks):
            c = self.cursor["chunk"]
            if c != self.cache_index:
                self.cache = read_chunk(self.root, self.chunks[c])
                self.cache_index = c
            r, offset = self.cursor["record"], self.cursor["offset"]
            if r >= len(self.cache):
                self.cursor = {"chunk": c + 1, "record": 0, "offset": 0}
                continue
            ids = self.cache[r]["qwen_ids" if self.large else "ids"]
            if offset >= len(ids):
                self.cursor.update(record=r + 1, offset=0)
                continue
            y = ids[offset : offset + min(512, limit)]
            prefix = ids[offset - 1] if offset else (ids[-1] if self.large else 1)
            x = [prefix] + y[:-1]
            self.cursor["offset"] += len(y)
            return x, y
        return None


def run(root, name, segment_tokens, commit):
    import hashlib
    import shutil

    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoTokenizer

    from distillation.curriculum import training_text, verify
    from distillation.curriculum_train import make_windows

    torch.manual_seed(42)
    torch.set_num_threads(4)
    root = Path(root)
    output = root / name
    output.mkdir(exist_ok=True)
    if not (output / "source").exists():
        shutil.copytree(
            "/project/distillation",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__", "results"),
        )
        for package in ("model_10m", "model_100m"):
            shutil.copytree(
                Path("/project") / package,
                output / package,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
    large = name == "existing-216m"
    target = 4_500_000 if large else 500_000_000
    latest = output / "latest.pt"
    saved = (
        torch.load(latest, map_location="cpu", weights_only=False)
        if latest.exists()
        else None
    )
    if large:
        parent = saved or torch.load(
            Path("/distill") / LARGE_PARENT, map_location="cpu", weights_only=False
        )
        config = ModelConfig(**parent["model_config"])
        tokenizer = AutoTokenizer.from_pretrained(
            Path("/distill") / LARGE_PARENT.split("/")[0] / "tokenizer"
        )
        tokenizer_hash = hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest()
        bos = tokenizer.eos_token_id
    else:
        layers, heads, width = (8, 6, 384) if name == "fresh-27m" else (10, 8, 512)
        config = ModelConfig(
            vocab_size=32768, n_layer=layers, n_head=heads, n_embd=width, block_size=512
        )
        tokenizer = load_frozen()
        tokenizer_hash, bos = TOKENIZER_SHA256, 1
    settings = {
        "name": name,
        "target": target,
        "corpus": str(root),
        "tokenizer_sha256": tokenizer_hash,
        "objective": "all-token CE sequence distillation",
        "lr": 3e-5 if large else 3e-4,
        "warmup_tokens": 100_000 if large else 1_000_000,
        "initialization": LARGE_PARENT if large else "random seed 42",
        "context": 512,
    }
    if saved and saved["settings"] != settings:
        raise ValueError("Resume configuration mismatch")
    model = GPT(config)
    if saved or large:
        model.load_state_dict((saved or parent)["model"], strict=True)
    model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    tokens, step = (saved["tokens"], saved["step"]) if saved else (0, 0)
    stream = Stream(root, saved["cursor"] if saved else None, large=large)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    if large:
        del parent
    del saved
    atomic_json(output / "request.json", settings)
    corpus = Path("/distill/qwen8b-python-curriculum-20261021-02")
    validation = [
        r
        for r in map(json.loads, (corpus / "accepted.jsonl").read_text().splitlines())
        if r["split"] == "validation"
    ]
    tasks = [
        r
        for r in map(json.loads, (corpus / "tasks.jsonl").read_text().splitlines())
        if r["split"] == "validation"
    ]
    if large:
        val = []
        for record in validation:
            ids = (
                [bos]
                + tokenizer.encode(training_text(record), add_special_tokens=False)
                + [bos]
            )
            for offset in range(0, len(ids) - 1, 512):
                block = ids[offset : offset + 513]
                val.append((block[:-1], block[1:]))
    else:
        val = make_windows(validation, tokenizer)

    def tensors(batch):
        length = max(len(y) for _, y in batch)
        x = torch.full((len(batch), length), bos, device="cuda", dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (a, b) in enumerate(batch):
            x[i, : len(a)] = torch.tensor(a, device="cuda")
            y[i, : len(b)] = torch.tensor(b, device="cuda")
        return x, y

    def evaluate(generations=False):
        model.eval()
        loss = count = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(val), 4):
                x, y = tensors(val[start : start + 4])
                loss += F.cross_entropy(
                    model(x).float().flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                count += (y != -100).sum().item()
        result = {
            "tokens": tokens,
            "step": step,
            "validation_ce": loss / count,
            "validation_tokens": count,
        }
        if generations:
            outputs, seen = [], set()
            for task in tasks:
                if task["family"] in seen:
                    continue
                seen.add(task["family"])
                prompt = f"### Problem\n{task['prompt']}\n### Python\n"
                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                ids = [bos] + (encoded if large else encoded.ids)
                generated = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(192):
                        token = (
                            model(torch.tensor([ids[-512:]], device="cuda"))[0, -1]
                            .argmax()
                            .item()
                        )
                        if token == (bos if large else 2):
                            break
                        generated.append(token)
                        ids.append(token)
                text = tokenizer.decode(generated, skip_special_tokens=False)
                result_code = verify(
                    text.split("### Explanation", 1)[0].strip(), task["cases"]
                )
                outputs.append(
                    {
                        "prompt": prompt,
                        "family": task["family"],
                        "text": text,
                        "verification": result_code,
                    }
                )
            result["diagnostic_passes"] = sum(
                r["verification"]["passed"] for r in outputs
            )
            atomic_json(output / f"generations-{tokens:012d}.json", outputs)
        with (output / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps({"name": name, **result}), flush=True)
        model.train()
        return result

    def checkpoint():
        temporary = output / "latest.partial"
        torch.save(
            {
                "format": "scale-sequence-v1",
                "settings": settings,
                "model_config": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "tokens": tokens,
                "step": step,
                "cursor": stream.cursor,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            temporary,
        )
        temporary.replace(latest)
        state = {
            "name": name,
            "tokens": tokens,
            "target": target,
            "step": step,
            "parameters": sum(p.numel() for p in model.parameters()),
            "status": "completed" if tokens == target else "waiting_for_next_segment",
        }
        atomic_json(output / "progress.json", state)
        commit()
        return state

    initial = tokens
    stop = min(target, tokens + segment_tokens)
    started = time.monotonic()
    if tokens == 0:
        evaluate(generations=True)
        checkpoint()
    next_eval = (
        ((tokens // 1_000_000) + 1) * 1_000_000
        if large
        else ((tokens // 10_000_000) + 1) * 10_000_000
    )
    last_checkpoint = tokens
    while tokens < stop and time.monotonic() - started < 72000:
        boundary = min(stop, next_eval)
        batches, count = [], 0
        for _ in range(4):
            batch = []
            for _ in range(4 if large else 8):
                if count >= boundary - tokens:
                    break
                item = stream.next(min(512, boundary - tokens - count))
                if item is None:
                    break
                batch.append(item)
                count += len(item[1])
            if batch:
                batches.append(batch)
        if not count:
            break
        optimizer.zero_grad(set_to_none=True)
        fraction = min(1.0, (tokens + count) / settings["warmup_tokens"])
        decay = (
            0.1
            + 0.9 * (1 + math.cos(math.pi * min(1.0, (tokens + count) / target))) / 2
        )
        for group in optimizer.param_groups:
            group["lr"] = settings["lr"] * fraction * decay
        total_loss = 0.0
        for batch in batches:
            x, y = tensors(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (
                    F.cross_entropy(
                        model(x).float().flatten(0, 1),
                        y.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    / count
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        tokens += count
        step += 1
        if step % 20 == 0 or tokens == boundary:
            metric = {
                "name": name,
                "tokens": tokens,
                "step": step,
                "ce": total_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "segment_tokens_per_second": (tokens - initial)
                / (time.monotonic() - started),
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)
        if tokens == next_eval:
            evaluate(generations=large or tokens % 50_000_000 == 0)
            next_eval += 1_000_000 if large else 10_000_000
        if tokens - last_checkpoint >= 2_000_000:
            checkpoint()
            last_checkpoint = tokens
    evaluate(generations=tokens == target)
    result = checkpoint()
    # Retain a weights-only checkpoint at every completed segment.
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(config),
            "tokens": tokens,
            "settings": settings,
        },
        output / f"weights-{tokens:012d}.pt",
    )
    commit()
    return result
