"""Bounded online probability distillation on one GPU, with durable resume."""

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from distillation.alignment import (
    byte_probabilities,
    marginal_kl,
    shared_boundaries,
    vocabulary,
)
from distillation.data import CodeWindows


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
    """Train from `parents/parent` (or resume `runs/resume`) into `output`.

    `manifest` and `database` are absolute paths of a code snapshot inventory and its
    code-only SQLite database. The teacher is a Hugging Face repo pinned to `revision`.
    """
    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    context = 256
    settings = {
        "teacher": teacher_repo,
        "revision": revision,
        "context": context,
        "alpha": 0.5,
        "lr": 1e-5,
        "temperature": 1.0,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": str(database),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    }
    source_hashes = {
        str(p.relative_to("/project")): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["distillation", "model_100m", "model_10m"]
        for p in (Path("/project") / folder).rglob("*.py")
    }
    (output / "request.json").write_text(
        json.dumps(
            dict(
                settings,
                source_sha256=source_hashes,
                target_tokens=target_tokens,
                max_seconds=max_seconds,
                parent=parent,
                resume=resume,
            ),
            indent=2,
        )
    )
    for folder in ["distillation", "model_100m", "model_10m"]:
        shutil.copytree(
            Path("/project") / folder,
            output / "source" / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copyfile(manifest, output / "data-manifest.json")
    commit()
    tokenizer = load_frozen()
    student_pieces, student_buckets = vocabulary(tokenizer.to_str(), 32768)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_repo, revision=revision)
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
    teacher_pieces, teacher_buckets = vocabulary(
        teacher_tokenizer.backend_tokenizer.to_str(), teacher.config.vocab_size
    )
    config = ModelConfig(vocab_size=32768, block_size=context)
    student = GPT(config).cuda()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=settings["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    checkpoint_input = f"{runs}/{resume}" if resume else f"{parents}/{parent}"
    saved = torch.load(checkpoint_input, map_location="cpu", weights_only=False)
    if resume:
        if saved["settings"] != settings or saved["source_sha256"] != source_hashes:
            raise ValueError("Resume settings/source mismatch")
    else:
        if saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Parent tokenizer mismatch")
        if any(
            saved["config"][k] != v
            for k, v in [("n_layer", 14), ("n_head", 12), ("n_embd", 768)]
        ):
            raise ValueError("Parent architecture mismatch")
    student.load_state_dict(saved["model"], strict=True)
    parent_hash = hashlib.sha256()
    with open(checkpoint_input, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            parent_hash.update(chunk)
    tokens, step, aligned, skipped = 0, 0, 0, 0
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        tokens, step, aligned, skipped = (
            saved[k] for k in ["tokens", "step", "aligned", "skipped"]
        )
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    data = CodeWindows(
        manifest, database, tokenizer, context, state=saved["data"] if resume else None
    )
    validation = CodeWindows(manifest, database, tokenizer, context, split="validation")
    validation_windows = []
    for _ in range(16):
        try:
            validation_windows.append(validation.next())
        except StopIteration:
            break
    if not validation_windows:
        raise ValueError("No held-out code windows")
    del saved

    def evaluate():
        student.eval()
        loss, count = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in validation_windows:
                logits = student(torch.tensor([x], device="cuda"))
                loss += F.cross_entropy(
                    logits.float().flatten(0, 1),
                    torch.tensor(y, device="cuda"),
                    reduction="sum",
                ).item()
                count += len(y)
        student.train()
        return {"ce": loss / count, "tokens": count, "windows": len(validation_windows)}

    initial_validation = evaluate()
    session_start_tokens = tokens
    training_started = time.monotonic()
    stop = "token_limit"
    last = {}
    latest = None

    def checkpoint():
        nonlocal latest
        path = output / f"step-{step:09d}.pt"
        if path.exists():
            return
        temp = path.with_suffix(".partial")
        torch.save(
            {
                "format": "byte-distillation-v1",
                "model": student.state_dict(),
                "model_config": asdict(config),
                "optimizer": optimizer.state_dict(),
                "settings": settings,
                "source_sha256": source_hashes,
                "data": data.state,
                "tokens": tokens,
                "step": step,
                "aligned": aligned,
                "skipped": skipped,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "parent": resume or parent,
                "parent_sha256": parent_hash.hexdigest(),
            },
            temp,
        )
        temp.replace(path)
        latest = str(path)
        (output / "latest.json").write_text(
            json.dumps({"checkpoint": latest, "tokens": tokens})
        )
        commit()

    while tokens < target_tokens:
        if time.monotonic() - started > max_seconds:
            stop = "time_limit"
            break
        cursor_before = dict(data.state)
        try:
            x, y = data.next()
        except StopIteration:
            stop = "source_exhausted"
            break
        # A partial final window would complicate cursor semantics; stop before it.
        if tokens + len(y) > target_tokens:
            data.state = cursor_before
            stop = "token_limit"
            break
        raw = b"".join(student_pieces[t] or b"" for t in x)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped += 1
            continue
        normalizer = teacher_tokenizer.backend_tokenizer.normalizer
        if normalizer is not None and normalizer.normalize_str(text) != text:
            skipped += 1
            continue
        tids = teacher_tokenizer.encode(text, add_special_tokens=False)
        if not tids:
            skipped += 1
            continue
        if len(tids) > 2048:
            raise ValueError("Teacher context exceeds bounded pilot limit")
        si, ti = shared_boundaries(x, tids, student_pieces, teacher_pieces)
        if not si:
            skipped += 1
            continue
        with torch.no_grad():
            teacher_logits = teacher(
                torch.tensor([tids], device="cuda"), use_cache=False
            ).logits[0]
            targets = byte_probabilities(teacher_logits[ti], teacher_buckets)
            del teacher_logits
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student(torch.tensor([x], device="cuda"))[0]
            ce = F.cross_entropy(logits.float(), torch.tensor(y, device="cuda"))
            kd = marginal_kl(logits[si], targets, student_buckets)
            loss = (1 - settings["alpha"]) * ce + settings["alpha"] * kd
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        tokens += len(y)
        aligned += len(si)
        step += 1
        last = {
            "step": step,
            "tokens": tokens,
            "ce": ce.item(),
            "byte_kl": kd.item(),
            "aligned": len(si),
            "grad_norm": norm.item(),
        }
        if step % 20 == 0:
            print(json.dumps(last), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(last) + "\n")
        if step % 500 == 0:
            checkpoint()
    checkpoint()
    elapsed = time.monotonic() - training_started
    report = dict(
        settings,
        parent=resume or parent,
        parent_sha256=parent_hash.hexdigest(),
        initial_validation=initial_validation,
        final_validation=evaluate(),
        tokens=tokens,
        session_tokens=tokens - session_start_tokens,
        steps=step,
        aligned_boundaries=aligned,
        skipped_unicode_or_empty_windows=skipped,
        train_seconds=elapsed,
        loss_tokens_per_second=(tokens - session_start_tokens) / elapsed,
        elapsed_seconds=time.monotonic() - started,
        stop_reason=stop,
        last_metrics=last,
        checkpoint=latest,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in student.parameters()),
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    commit()
    return report
