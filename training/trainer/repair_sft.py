"""Bounded single-GPU repair SFT, starting from a trusted PR #26 checkpoint."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from training_data.repair_sft import FORMAT, prepare
from training_data.subword import TOKENIZER_SHA256, digest, load_frozen


@dataclass(frozen=True)
class Settings:
    steps: int = 200
    batch_size: int = 4
    context: int = 1024
    lr: float = 1e-5
    seed: int = 42
    evaluate_every: int = 50
    save_every: int = 50
    max_seconds: int = 1200

    def __post_init__(self):
        for key in (
            "steps",
            "batch_size",
            "context",
            "evaluate_every",
            "save_every",
            "max_seconds",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1e-3:
            raise ValueError("Invalid learning rate")
        if self.max_seconds > 3000:
            raise ValueError("Pilot capped at 3000 seconds")


def collate(rows, device):
    import torch

    width = max(len(row["x"]) for row in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        x[i, : len(row["x"])] = torch.tensor(row["x"], device=device)
        y[i, : len(row["y"])] = torch.tensor(row["y"], device=device)
    return x, y


def train(parent, dataset, output, settings, commit=lambda: None, resume=None):
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Training is CUDA/Modal only")
    tokenizer = load_frozen()
    splits, data_hash = prepare(dataset, tokenizer, settings.context)
    # Only load our trusted private run-volume checkpoints, never dataset-supplied files.
    base = torch.load(parent, map_location="cpu", weights_only=False)
    if base.get("format") != "scholarly-ddp-v1":
        raise ValueError("Expected distributed pretraining parent")
    cfg = base["config"]
    if settings.context > cfg["context"]:
        raise ValueError("SFT context exceeds parent context")
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    rng = random.Random(settings.seed)
    model = GPT(
        ModelConfig(
            vocab_size=32768,
            block_size=cfg["context"],
            **{k: cfg[k] for k in ("n_layer", "n_head", "n_embd")},
        )
    ).cuda()
    model.load_state_dict(base["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=0.01)
    source_hashes = {"trainer": digest(__file__)}
    import model_10m.model as shared_module
    import model_100m.model as model_module
    import training_data.repair_sft as data_module

    source_hashes.update(
        data=digest(data_module.__file__),
        model=digest(model_module.__file__),
        shared_model=digest(shared_module.__file__),
    )
    identity = {
        "dataset_sha256": data_hash,
        "parent_sha256": digest(parent),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "settings": asdict(settings),
        "source_sha256": source_hashes,
    }
    if base["source_sha256"]["model"] != source_hashes["model"]:
        raise ValueError("Model implementation differs from parent checkpoint")
    step, answer_tokens = 0, 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT or saved["identity"] != identity:
            raise ValueError("SFT resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, answer_tokens = saved["step"], saved["answer_tokens"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(identity, indent=2))
    # Preserve exact inputs and executable sources for reproducibility.
    import shutil

    shutil.copyfile(dataset, output / "examples.jsonl")
    from training_data.subword import DEFAULT_FOLDER

    shutil.copytree(DEFAULT_FOLDER, output / "tokenizer")
    for name, module in (
        ("trainer", __file__),
        ("data", data_module.__file__),
        ("model", model_module.__file__),
        ("shared_model", shared_module.__file__),
    ):
        shutil.copyfile(module, output / f"{name}.py")
    del base

    def loss(rows):
        x, y = collate(rows, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = F.cross_entropy(
                model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
            )
        return value, (y != -100).sum()

    def evaluate():
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for row in splits["validation"]:
                value, n = loss([row])
                total += value.item()
                count += n.item()
        model.train()
        return total / count

    def record(**values):
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"step": step, "answer_tokens": answer_tokens, **values}) + "\n"
            )

    def save():
        target = output / f"step-{step:09d}.pt"
        temporary = target.with_suffix(".partial")
        torch.save(
            {
                "format": FORMAT,
                "identity": identity,
                "model_config": asdict(model.config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "answer_tokens": answer_tokens,
                "sampling_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            temporary,
        )
        temporary.replace(target)
        commit()
        return str(target)

    started = time.monotonic()
    record(validation_loss=evaluate(), phase="baseline")
    last_saved = -1
    while step < settings.steps and time.monotonic() - started < settings.max_seconds:
        rows = [rng.choice(splits["train"]) for _ in range(settings.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        value, count = loss(rows)
        objective = value / count
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite SFT loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        answer_tokens += count.item()
        record(train_loss=objective.item())
        if step % settings.evaluate_every == 0:
            record(validation_loss=evaluate())
        if step % settings.save_every == 0:
            save()
            last_saved = step
    record(validation_loss=evaluate(), phase="final")
    path = str(output / f"step-{step:09d}.pt") if last_saved == step else save()
    commit()
    return {"checkpoint": path, "step": step, "answer_tokens": answer_tokens}
