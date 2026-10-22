"""50M-token direct-KL run; validation regression is observed, not a stop rule."""

import hashlib
import json
import shutil
import signal
import time
from dataclasses import asdict
from pathlib import Path

from distillation.evaluate import select_windows
from distillation.shared import (
    SharedCodeWindows,
    take_batch,
    token_losses,
    transfer_embeddings,
)

MILESTONES = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]


def run(
    output,
    commit,
    *,
    teacher_repo,
    revision,
    parent,
    manifest,
    database,
    target_tokens,
    max_seconds,
    resume="",
    parents="/runs",
    runs="/distill",
):
    """Shared-tokenizer KL from `parents/parent` (or resume `runs/resume`) into `output`.

    Validation and checkpoints also happen at the fixed MILESTONES below the target.
    """
    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(teacher_repo, revision=revision)
    frozen = load_frozen()
    tokenizer_hash = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest()
    settings = {
        "teacher": teacher_repo,
        "revision": revision,
        "tokenizer_sha256": tokenizer_hash,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "context": 512,
        "microbatch": 4,
        "accumulation": 4,
        "alpha": 0.5,
        "temperature": 1.0,
        "lr": 1e-5,
        "embedding_lr": 1e-4,
        "warmup_tokens": 1_000_000,
        "target_tokens": target_tokens,
        "parent": parent,
        "seed": 42,
    }
    sources = {
        str(p.relative_to("/project")): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["distillation", "model_100m", "model_10m"]
        for p in (Path("/project") / folder).rglob("*.py")
    }
    for folder in ["distillation", "model_100m", "model_10m"]:
        shutil.copytree(
            Path("/project") / folder,
            output / "source" / folder,
            ignore=shutil.ignore_patterns("__pycache__", "artifacts"),
        )
    shutil.copyfile(manifest, output / "data-manifest.json")
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "request.json").write_text(
        json.dumps(
            dict(
                settings, source_sha256=sources, resume=resume, max_seconds=max_seconds
            ),
            indent=2,
        )
    )
    commit()
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            teacher_repo,
            revision=revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    eos = teacher.config.eos_token_id
    config = ModelConfig(
        vocab_size=teacher.config.vocab_size, block_size=settings["context"]
    )
    student = GPT(config)
    checkpoint_input = f"{runs}/{resume}" if resume else f"{parents}/{parent}"
    saved = torch.load(checkpoint_input, map_location="cpu", weights_only=False)
    if resume:
        if saved["settings"] != settings or saved["source_sha256"] != sources:
            raise ValueError("Resume settings/source mismatch; restore archived source")
        student.load_state_dict(saved["model"], strict=True)
        initialization = saved["initialization"]
    else:
        if saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Parent tokenizer mismatch")
        backbone = {
            k: v
            for k, v in saved["model"].items()
            if k not in ("token_embedding.weight", "lm_head.weight")
        }
        missing = student.load_state_dict(backbone, strict=False)
        if (
            set(missing.missing_keys) != {"token_embedding.weight", "lm_head.weight"}
            or missing.unexpected_keys
        ):
            raise ValueError("Parent backbone mismatch")
        initialization = transfer_embeddings(
            saved["model"]["token_embedding.weight"],
            frozen,
            tokenizer,
            student.token_embedding.weight,
            eos,
        )
    parent_hash = hashlib.sha256()
    with open(checkpoint_input, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            parent_hash.update(chunk)
    student.cuda().train()
    embedding = student.token_embedding.weight
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in student.parameters() if p is not embedding],
                "lr": settings["lr"],
            },
            {"params": [embedding], "lr": settings["embedding_lr"]},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    tokens = step = 0
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        tokens, step = saved["tokens"], saved["step"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    data = SharedCodeWindows(
        manifest,
        database,
        tokenizer,
        context=settings["context"],
        frozen_tokenizer=frozen,
        eos_id=eos,
        state=saved["data"] if resume else None,
    )
    del saved
    validation = SharedCodeWindows(
        manifest,
        database,
        tokenizer,
        context=settings["context"],
        split="validation",
        frozen_tokenizer=frozen,
        eos_id=eos,
    )
    windows = select_windows(validation, max_documents=128, windows_per_document=4)
    validation_hash = hashlib.sha256(
        json.dumps(windows, separators=(",", ":")).encode()
    ).hexdigest()
    stop_requested = False

    def stop_handler(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    def tensors(batch):
        length = max(len(y) for _, y in batch)
        x = torch.full((len(batch), length), eos, device="cuda", dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (a, b) in enumerate(batch):
            x[i, : len(a)] = torch.tensor(a, device="cuda")
            y[i, : len(b)] = torch.tensor(b, device="cuda")
        return x, y

    def evaluate():
        student.eval()
        loss = count = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(0, len(windows), 4):
                x, y = tensors([(a, b) for _, a, b in windows[offset : offset + 4]])
                logits = student(x)
                loss += F.cross_entropy(
                    logits.float().flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                count += (y != -100).sum().item()
        student.train()
        report = {
            "training_tokens": tokens,
            "step": step,
            "ce": loss / count,
            "tokens": count,
            "documents": len({key for key, _, _ in windows}),
            "windows": len(windows),
            "window_sha256": validation_hash,
        }
        with (output / "validation.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        print(json.dumps({"validation": report}), flush=True)
        return report

    latest = None

    def checkpoint():
        nonlocal latest
        path = output / f"step-{step:09d}.pt"
        if path.exists():
            return
        temp = path.with_suffix(".partial")
        torch.save(
            {
                "format": "shared-token-distillation-v1",
                "settings": settings,
                "source_sha256": sources,
                "model_config": asdict(config),
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "tokens": tokens,
                "step": step,
                "data": dict(data.state),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "initialization": initialization,
                "parent": checkpoint_input,
                "parent_sha256": parent_hash.hexdigest(),
                "validation_hash": validation_hash,
            },
            temp,
        )
        temp.replace(path)
        latest = str(path)
        (output / "latest.json").write_text(
            json.dumps({"checkpoint": latest, "tokens": tokens, "step": step})
        )
        commit()

    initial = evaluate()
    checkpoint()
    print(
        json.dumps(
            {
                "ready": True,
                "parameters": sum(p.numel() for p in student.parameters()),
                "initialization": initialization,
                "target_tokens": target_tokens,
            }
        ),
        flush=True,
    )
    train_started = time.monotonic()
    start_tokens = tokens
    next_periodic = ((tokens // 5_000_000) + 1) * 5_000_000
    stop = "token_limit"
    last = {}
    while tokens < target_tokens:
        if stop_requested or time.monotonic() - started >= max_seconds:
            stop = "signal" if stop_requested else "time_limit"
            break
        boundary = min(
            [x for x in MILESTONES if x > tokens] + [target_tokens, next_periodic]
        )
        boundary = min(boundary, target_tokens)
        # Preserve the exact window cursor at milestone boundaries, including tails.
        batches = []
        count = 0
        for _ in range(settings["accumulation"]):
            batch = take_batch(data, settings["microbatch"], boundary - tokens - count)
            if not batch:
                break
            batches.append(batch)
            count += sum(len(y) for _, y in batch)
            if tokens + count == boundary:
                break
        if not batches:
            stop = "source_exhausted"
            break
        optimizer.zero_grad(set_to_none=True)
        scale = min(1.0, (tokens + count) / settings["warmup_tokens"])
        for group, lr in zip(
            optimizer.param_groups,
            [settings["lr"], settings["embedding_lr"]],
            strict=True,
        ):
            group["lr"] = lr * scale
        ce_sum = kl_sum = 0.0
        for batch in batches:
            x, y = tensors(batch)
            with torch.no_grad():
                t = teacher(
                    x, attention_mask=(y != -100).long(), use_cache=False
                ).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = student(x)
                ce, kl = token_losses(s, t, y)
                loss = ((1 - settings["alpha"]) * ce + settings["alpha"] * kl) / count
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite distillation loss")
            loss.backward()
            ce_sum += ce.item()
            kl_sum += kl.item()
            del s, t, loss, ce, kl
        norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        tokens += count
        step += 1
        elapsed = time.monotonic() - train_started
        last = {
            "step": step,
            "tokens": tokens,
            "ce": ce_sum / count,
            "token_kl": kl_sum / count,
            "grad_norm": norm.item(),
            "tokens_per_second": (tokens - start_tokens) / elapsed,
            "lr": optimizer.param_groups[0]["lr"],
            "embedding_lr": optimizer.param_groups[1]["lr"],
        }
        if step % 10 == 0 or tokens == boundary:
            print(json.dumps(last), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(last) + "\n")
        if tokens == boundary:
            evaluate()
            checkpoint()
            if tokens >= next_periodic:
                next_periodic += 5_000_000
            # A truncated window's unused suffix must be re-read from the saved cursor.
            data.iterator = data._windows()
        elif step % 500 == 0:
            checkpoint()
    checkpoint()
    final = evaluate()
    report = {
        "settings": settings,
        "tokens": tokens,
        "steps": step,
        "stop_reason": stop,
        "initial_validation": initial,
        "final_validation": final,
        "last_metrics": last,
        "checkpoint": latest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "parameters": sum(p.numel() for p in student.parameters()),
        "initialization": initialization,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    commit()
    return report
