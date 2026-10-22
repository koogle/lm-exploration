"""Bounded single-GPU repair SFT (ordinary or CIG-weighted) from a trusted pretraining checkpoint.

The parent is an explicit `scholarly-ddp-v1` checkpoint on the private run volume; launchers
pass it in. See `sft/README.md` for the data contract and launch commands.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from training_data.repair_sft import FORMAT, prepare
from training_data.subword import LOSS_IGNORE_INDEX, TOKENIZER_SHA256, digest, load_frozen


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
    objective: str = "sft"
    cig_proximity: float = 0.3

    def __post_init__(self):
        if self.objective not in ("sft", "cig"):
            raise ValueError("Unknown SFT objective")
        if not math.isfinite(self.cig_proximity) or self.cig_proximity < 0.01:
            raise ValueError("Invalid CIG proximity")
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
        if self.max_seconds > 14400:
            raise ValueError("SFT capped at four hours per invocation")


# Sources of the recorded 200-update CIG/SFT pilots (PR #29). `extend=True` may migrate
# checkpoints written by these reviewed predecessors or by identical current source only.
PILOT_TRAINER_SHA256 = "b2b8ceebde172b280fcb5ec7ddbb86e5df53c80ad9406087be9704d3e869656b"
PILOT_CIG_SHA256 = "fc241179be318316f05ad2d37a41eb32f19f33f31293a29c7c2229a8f070152d"
REVIEWED_SOURCES = {"trainer": PILOT_TRAINER_SHA256, "cig": PILOT_CIG_SHA256}


def validate_resume_identity(saved, current, step, extend=False):
    """Permit explicit run-length extensions only; retain all learning/data identity."""
    import copy

    if not extend:
        if saved != current:
            raise ValueError("SFT resume identity mismatch")
        return
    if current["settings"]["steps"] <= max(step, saved["settings"]["steps"]):
        raise ValueError("Extension must increase the total step target")
    adjusted = copy.deepcopy(saved)
    for key in ("steps", "save_every", "evaluate_every", "max_seconds"):
        adjusted["settings"][key] = current["settings"][key]
    for name, reviewed in REVIEWED_SOURCES.items():
        previous = adjusted["source_sha256"][name]
        if previous not in (reviewed, current["source_sha256"][name]):
            raise ValueError(f"Unreviewed {name} source migration")
        adjusted["source_sha256"][name] = current["source_sha256"][name]
    if adjusted != current:
        raise ValueError("Extension changed model/data/objective/optimizer identity")


def collate(rows, device):
    import torch

    width = max(len(row["x"]) for row in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full_like(x, LOSS_IGNORE_INDEX)
    for i, row in enumerate(rows):
        x[i, : len(row["x"])] = torch.tensor(row["x"], device=device)
        y[i, : len(row["y"])] = torch.tensor(row["y"], device=device)
    return x, y


def train(
    parent,
    dataset,
    output,
    settings,
    commit=lambda: None,
    resume=None,
    stop_after_step=None,
    extend=False,
    deadline=None,
):
    """Train until the step target, `settings.max_seconds` or the absolute `deadline`.

    `deadline` is a Unix time from the launcher's 24-hour launch budget; the loop stops
    taking updates once it passes, then evaluates and saves a final checkpoint.
    """
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F

    if extend and resume is None:
        raise ValueError("Extension requires a resume checkpoint")
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing SFT work")
    if not torch.cuda.is_available():
        raise RuntimeError("Training is CUDA/Modal only")
    if stop_after_step is not None and not 1 <= stop_after_step <= settings.steps:
        raise ValueError("Invalid qualification stop step")
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
    from trainer import cig

    source_hashes = {"trainer": digest(__file__), "cig": digest(cig.__file__)}
    cache = {}
    if settings.objective == "cig":
        raw_rows = [
            json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()
        ]
        cache = cig.reference_cache(model, splits, raw_rows, tokenizer, settings.context)
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
        if saved.get("format") != FORMAT:
            raise ValueError("SFT resume format mismatch")
        validate_resume_identity(saved["identity"], identity, saved["step"], extend=extend)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, answer_tokens = saved["step"], saved["answer_tokens"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(identity, indent=2))
    if resume:
        (output / "continuation.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(resume),
                    "sha256": digest(resume),
                    "extended": extend,
                    "restored_step": step,
                    "restored_answer_tokens": answer_tokens,
                    "previous_identity": saved["identity"],
                },
                indent=2,
            )
        )
        del saved
    # Preserve exact inputs and executable sources for reproducibility.
    import shutil

    shutil.copyfile(dataset, output / "examples.jsonl")
    from training_data.subword import DEFAULT_FOLDER

    shutil.copytree(DEFAULT_FOLDER, output / "tokenizer")
    for name, module in (
        ("trainer", __file__),
        ("cig", cig.__file__),
        ("data", data_module.__file__),
        ("model", model_module.__file__),
        ("shared_model", shared_module.__file__),
    ):
        shutil.copyfile(module, output / f"{name}.py")
    if cache:
        torch.save(cache, output / "cig-reference.pt")
        values = torch.cat([v[v > 0] for v in cache.values()])
        (output / "cig-summary.json").write_text(
            json.dumps(
                {
                    "positive_tokens": values.numel(),
                    "answer_tokens": sum(
                        sum(t != LOSS_IGNORE_INDEX for t in r["y"]) for r in splits["train"]
                    ),
                    "mean_positive_information": values.mean().item() if values.numel() else 0.0,
                },
                indent=2,
            )
        )
    del base

    def loss(rows, training=False):
        x, y = collate(rows, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            if training and settings.objective == "cig":
                information = torch.zeros_like(y, dtype=torch.float32)
                for i, row in enumerate(rows):
                    information[i, : len(row["y"])] = cache[row["id"]].to("cuda")
                return cig.weighted_loss(logits, y, information, settings.cig_proximity)
            value = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
            )
        return value, (y != LOSS_IGNORE_INDEX).sum()

    def evaluate():
        model.eval()
        totals = {}
        with torch.inference_mode():
            for row in splits["validation"]:
                value, n = loss([row])
                bucket = totals.setdefault(row["language"], [0.0, 0])
                bucket[0] += value.item()
                bucket[1] += n.item()
        model.train()
        return {
            "validation_loss": sum(v[0] for v in totals.values())
            / sum(v[1] for v in totals.values()),
            "validation_by_language": {
                name: {"loss": total / count, "answer_tokens": count}
                for name, (total, count) in sorted(totals.items())
            },
        }

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
    initial_step = step
    record(**evaluate(), phase="baseline")
    print(json.dumps({"stage": "training", "step": step, "target": settings.steps}), flush=True)
    last_saved = -1
    target_step = settings.steps if stop_after_step is None else stop_after_step
    while (
        step < target_step
        and time.monotonic() - started < settings.max_seconds
        and (deadline is None or time.time() < deadline)
    ):
        rows = [rng.choice(splits["train"]) for _ in range(settings.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        value, count = loss(rows, training=True)
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
            metrics = evaluate()
            record(**metrics)
            print(json.dumps({"step": step, **metrics}), flush=True)
        if step % settings.save_every == 0:
            save()
            last_saved = step
    record(**evaluate(), phase="final")
    path = str(output / f"step-{step:09d}.pt") if last_saved == step else save()
    commit()
    return {
        "checkpoint": path,
        "step": step,
        "answer_tokens": answer_tokens,
        "segment_updates": step - initial_step,
        "segment_seconds": time.monotonic() - started,
    }
