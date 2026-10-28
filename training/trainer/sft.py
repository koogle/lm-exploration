"""Bounded single-GPU supervised fine-tuning from a pretraining checkpoint.

Each step samples `batch_size` examples of the task's dataset (`TASKS`; see
`training_data.sft`): a prompt and a complete written answer. The model is trained by
teacher forcing to predict every answer token and EOS given the prompt and the answer so
far; prompt tokens carry no loss. The loss is the mean answer-token cross-entropy
(`objective: "sft"`), or for repair data optionally CIG-weighted (`objective: "cig"`,
`trainer/cig.py`). The model never generates during training. Validation reports the same
answer-token cross-entropy per bucket.

Optional pretraining replay (`"replay": true` and the `replay_*` settings, the shared
`trainer/replay_hook.py`) adds `replay_batch_size` windows of pretraining text to every
step, trained with the ordinary next-token loss: the step backpropagates
`(1 - replay_weight) * answer loss`, then the hook adds `replay_weight * replay loss`. The learning rate is constant, or warms up linearly over
`warmup_steps` and follows a cosine to `min_lr_ratio * lr` (`schedule: "cosine"`).

The parent is an explicit `scholarly-ddp-v1` checkpoint on the private run volume; launchers
pass it in. See `sft/README.md` for the data formats and launch commands.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from training_data.math_sft import MATH
from training_data.repair_sft import REPAIR
from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    digest,
    load_frozen,
)

from trainer import batches
from trainer.batches import collate
from trainer.replay_hook import ReplayHook, ReplaySettings

# The SFT dataset (`training_data.sft.SFTDataset`) of each task.
TASKS = {"repair": REPAIR, "math": MATH}


@dataclass(frozen=True)
class Settings(ReplaySettings):
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
    task: str = "repair"
    schedule: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0

    def __post_init__(self):
        if self.objective not in ("sft", "cig"):
            raise ValueError("Unknown SFT objective")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}")
        if self.objective == "cig" and self.task != "repair":
            raise ValueError("CIG references are defined for repair prompts only")
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
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("schedule must be constant or cosine")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be from 0 to below steps")
        if not 0 < self.min_lr_ratio <= 1 or (self.schedule == "constant") != (
            self.min_lr_ratio == 1 and self.warmup_steps == 0
        ):
            raise ValueError("Only a cosine schedule warms up or decays (min_lr_ratio in (0, 1])")
        self.check_replay()

    def learning_rate(self, step):
        """The rate for update `step` (0-based)."""
        if self.schedule == "constant":
            return self.lr
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return self.lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)


def validate_resume_identity(saved, current, step, extend=False):
    """Permit explicit run-length extensions only; retain all learning/data/source identity."""
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
    if adjusted != current:
        raise ValueError("Extension changed source/model/data/objective/optimizer identity")


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
    data_root=Path("/data"),
):
    """Train until the step target, `settings.max_seconds` or the absolute `deadline`.

    `data_root` holds `settings.replay_snapshot`.

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
    import sys

    data = TASKS[settings.task]
    data_module = sys.modules[type(data).__module__]
    tokenizer = load_frozen()
    splits, data_hash = data.prepare(dataset, tokenizer, settings.context)
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

    source_hashes = {
        "trainer": digest(__file__),
        "batches": digest(batches.__file__),
        "cig": digest(cig.__file__),
    }
    cache = {}
    if settings.objective == "cig":
        raw_rows = [
            json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()
        ]
        cache = cig.reference_cache(model, splits, raw_rows, tokenizer, settings.context)
    import model_10m.model as shared_module
    import model_100m.model as model_module
    import training_data.sft as sft_data_module

    replay = ReplayHook.attach(settings, data_root, cfg["context"])
    source_hashes.update(
        **{name: digest(file) for name, file in ReplayHook.SOURCES.items()},
        data=digest(data_module.__file__),
        sft_data=digest(sft_data_module.__file__),
        model=digest(model_module.__file__),
        shared_model=digest(shared_module.__file__),
    )
    identity = {
        "dataset_sha256": data_hash,
        "parent_sha256": digest(parent),
        "replay": replay.identity() if replay else None,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "settings": asdict(settings),
        "source_sha256": source_hashes,
    }
    if base["source_sha256"]["model"] != source_hashes["model"]:
        raise ValueError("Model implementation differs from parent checkpoint")
    step, answer_tokens = 0, 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("format") != data.FORMAT:
            raise ValueError("SFT resume format mismatch")
        validate_resume_identity(saved["identity"], identity, saved["step"], extend=extend)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["sampling_rng"])
        if replay:
            replay.load_state(saved["replay_state"])
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
        ("batches", batches.__file__),
        ("cig", cig.__file__),
        ("data", data_module.__file__),
        ("sft_data", sft_data_module.__file__),
        *ReplayHook.SOURCES.items(),
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
                bucket = totals.setdefault(row["bucket"], [0.0, 0])
                bucket[0] += value.item()
                bucket[1] += n.item()
            replayed = replay.validation(model) if replay else None
        model.train()
        metrics = {
            "validation_loss": sum(v[0] for v in totals.values())
            / sum(v[1] for v in totals.values()),
            "validation_by_bucket": {
                name: {"loss": total / count, "answer_tokens": count}
                for name, (total, count) in sorted(totals.items())
            },
        }
        if replayed:
            metrics["replay_validation_by_source"] = replayed
        return metrics

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
                "format": data.FORMAT,
                "identity": identity,
                "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
                "model_config": asdict(model.config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "answer_tokens": answer_tokens,
                "sampling_rng": rng.getstate(),
                "replay_state": replay.state() if replay else None,
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
        learning_rate = settings.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        value, count = loss(rows, training=True)
        answer_loss = value / count
        if not torch.isfinite(answer_loss):
            raise RuntimeError("Non-finite SFT loss")
        values = {"train_loss": answer_loss.item()}
        (answer_loss * (1 - settings.replay_weight if replay else 1)).backward()
        if replay:
            values["replay_loss"] = replay.backward(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        answer_tokens += count.item()
        record(**values, lr=learning_rate)
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
