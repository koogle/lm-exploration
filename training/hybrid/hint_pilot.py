"""N-gram hint pilot: the hybrid decoder trained with and without hints (hints.py).

Two arms share the model, data order, optimizer and schedule; one adds hints from
an index of its own earlier optimizer steps. Evaluation reads the first
validation rows with no hints, with hints from the training rows, and with hints
from longer prefixes of the training split (`corpus_tables`, which keeps only the
contexts that occur in those validation rows). Hints never come from validation
text, and a hint at position t reads only x_(<=t).

Functions here take explicit paths and write JSON metrics; modal_hint_pilot.py
runs them on Modal.
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import hints

SETTINGS = {
    "context": 2048,
    "rows_per_step": 64,
    "micro_rows": 8,
    "steps": 1900,
    "warmup_steps": 100,
    "final_lr_fraction": 0.1,
    "grad_clip": 1.0,
    "seed": 1234,
    "validation_rows": 512,
    "flush_steps": 16,
    "corpus_tokens": [500_000_000, 1_000_000_000, 2_000_000_000],
    # climbmix-code-v6 has pending audits; the 124M runs on it pass the same override.
    "allow_unreviewed": True,
}


def loader(manifest, settings, split):
    from training_data.raw_token_stream import RawTokenMixture

    return RawTokenMixture(
        manifest, settings["context"], split, allow_unreviewed=settings["allow_unreviewed"]
    )


def validation_rows(manifest, settings):
    """The first `validation_rows` packed validation rows, (inputs, labels) as numpy."""
    data = loader(manifest, settings, "validation")
    x, y = data.next_batch(settings["validation_rows"])
    return x.numpy(), y.numpy()


def model_config(settings, hinted):
    from training_data.subword import EOS_TOKEN_ID, VOCAB_SIZE

    from .model import HybridConfig

    return HybridConfig(
        vocab_size=VOCAB_SIZE, eos_id=EOS_TOKEN_ID, block_size=settings["context"], hints=hinted
    )


def build_model(settings, hinted):
    import torch

    from .model import HybridLM

    torch.manual_seed(settings["seed"])
    lm = HybridLM(model_config(settings, hinted)).cuda()
    # Blocks compile one at a time; hint gathers stay eager (their sizes vary).
    for block in lm.blocks:
        block.compile()
    if lm.mtp is not None:
        lm.mtp.block.compile()
    return lm


def learning_rate_factor(step, settings):
    """Linear warmup, then cosine decay to `final_lr_fraction`."""
    warmup, total = settings["warmup_steps"], settings["steps"]
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    floor = settings["final_lr_fraction"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def micro_batches(x, y, micro_rows):
    return [(x[i : i + micro_rows], y[i : i + micro_rows]) for i in range(0, len(x), micro_rows)]


def evaluate(lm, rows, settings, tables=None):
    """Mean validation cross-entropy (nats/token) and hint coverage; `tables` as in hints.lookup."""
    import torch
    from training_data.subword import EOS_TOKEN_ID

    inputs, labels = rows
    ce = count = 0.0
    found_any = []
    lm.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in micro_batches(inputs, labels, settings["micro_rows"]):
            found = None
            if tables is not None:
                found = hints.lookup(tables, x, EOS_TOKEN_ID)
                found_any.append(found.cells[y != hints.IGNORE_INDEX] > 0)
                found = found.to("cuda")
            losses = lm(torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), hints=found)
            ce += losses["ce"].item()
            count += losses["count"].item()
    lm.train()
    result = {"loss": ce / count, "tokens": int(count)}
    if found_any:
        result["hinted_share"] = float(np.concatenate([f.ravel() for f in found_any]).mean())
    return result


def train_arm(manifest, output, settings, hinted, deadline, reserve_seconds=1800, log=print):
    """Train one arm, evaluate it on the validation rows and save it under `output`.

    Training stops early, recording the steps reached, when `deadline` minus
    `reserve_seconds` (for evaluation and saving) passes.
    """
    import torch
    from training_data.subword import EOS_TOKEN_ID

    from trainer.optimizers import set_learning_rate

    from .documents import mtp_labels
    from .model import mtp_weight
    from .optim import HybridOptimizer, OptimizerConfig

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    lm = build_model(settings, hinted)
    config = OptimizerConfig()
    optimizer = HybridOptimizer(lm, config, fused=True)
    data = loader(manifest, settings, "train")
    index = hints.HintIndex(EOS_TOKEN_ID, settings["flush_steps"]) if hinted else None
    micro = settings["micro_rows"]

    def prepare(step):
        """The step's rows and, for the hinted arm, each micro batch's hints."""
        x, y = data.next_batch(settings["rows_per_step"])
        x, y = x.numpy(), y.numpy()
        found = through = None
        if index is not None:
            found, through = [], -1
            for xs, _ in micro_batches(x, y, micro):
                one, through = index.hints(step, xs)
                found.append(one)
        return x, y, found, through

    worker = ThreadPoolExecutor(max_workers=1)
    pending = worker.submit(prepare, 0)
    history, started, tokens = [], time.time(), 0
    steps_done = 0
    for step in range(settings["steps"]):
        if time.time() > deadline - reserve_seconds:
            log(json.dumps({"stopped_early_at_step": step}))
            break
        x, y, found, through = pending.result()
        if step + 1 < settings["steps"]:
            pending = worker.submit(prepare, step + 1)
        factor = learning_rate_factor(step, settings)
        set_learning_rate(optimizer, factor * config.adamw_lr)
        mtp = mtp_labels(torch.from_numpy(x), torch.from_numpy(y), EOS_TOKEN_ID)
        count = int((y != hints.IGNORE_INDEX).sum())
        mtp_count = int((mtp != hints.IGNORE_INDEX).sum())
        weight = mtp_weight(lm.config, step / settings["steps"])
        ce_sum = 0.0
        for i, (xs, ys) in enumerate(micro_batches(x, y, micro)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = lm(
                    torch.from_numpy(xs).cuda(non_blocking=True),
                    torch.from_numpy(ys).cuda(non_blocking=True),
                    mtp[i * micro : (i + 1) * micro].cuda(non_blocking=True),
                    hints=None if found is None else found[i].to("cuda"),
                )
            loss = losses["loss"] / count + weight * losses["mtp_loss"] / mtp_count
            loss.backward()
            ce_sum += losses["ce"].detach()
        norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), settings["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if index is not None:
            index.add(step, x, y)
        tokens += count
        steps_done = step + 1
        if step % 10 == 0 or steps_done == settings["steps"]:
            record = {
                "step": step,
                "ce": float(ce_sum) / count,
                "grad_norm": float(norm),
                "lr_factor": factor,
                "tokens": tokens,
                "tokens_per_second": tokens / (time.time() - started),
                "elapsed_seconds": time.time() - started,
            }
            if index is not None:
                coverage = [
                    f.cells[ys != hints.IGNORE_INDEX] > 0
                    for f, (_, ys) in zip(found, micro_batches(x, y, micro))
                ]
                record |= {
                    "hinted_share": float(np.concatenate([c.ravel() for c in coverage]).mean()),
                    "index_lag_steps": step - through,
                    "next_scale_mean": float(lm.hint.next_scale.mean()),
                    "site_scales": lm.hint.site_scales.tolist(),
                }
            history.append(record)
            log(json.dumps(record))
    worker.shutdown(wait=True, cancel_futures=True)

    rows = validation_rows(manifest, settings)
    evaluation = {"none": evaluate(lm, rows, settings)}
    if index is not None:
        index.flush()
        evaluation["training_rows"] = evaluate(lm, rows, settings, index.tables)
        keys = hints.query_keys([rows[0]], EOS_TOKEN_ID)
        own = {n: hints.Table.merge(index.tables[n]).filtered(keys[n]) for n in hints.LENGTHS}
        hints.save_tables(
            output / "training-rows-validation-tables.npz",
            own,
            rows=steps_done * settings["rows_per_step"],
        )
    torch.save(
        {
            "settings": settings,
            "hinted": hinted,
            "steps": steps_done,
            "model": {k: v.bfloat16() for k, v in lm.state_dict().items()},
        },
        output / "model.pt",
    )
    result = {
        "hinted": hinted,
        "settings": settings,
        "steps": steps_done,
        "training_tokens": tokens,
        "training_rows": steps_done * settings["rows_per_step"],
        "evaluation": evaluation,
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def corpus_tables(manifest, output, settings, rows_milestones, deadline, log=print):
    """Index prefixes of the training split for the validation rows' contexts only.

    Streams training rows in the trainer's order (`rows_per_step` at a time) and
    keeps a (context, next token) pair only when its context occurs in the
    validation rows. At each milestone (a row count) saves `corpus-<rows>.npz`.
    """
    from training_data.subword import EOS_TOKEN_ID

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = validation_rows(manifest, settings)
    keys = hints.query_keys([rows[0]], EOS_TOKEN_ID)
    data = loader(manifest, settings, "train")
    kept = {n: [] for n in hints.LENGTHS}
    done, saved, started = 0, [], time.time()
    milestones = sorted(rows_milestones)
    while milestones and time.time() < deadline:
        x, y = data.next_batch(settings["rows_per_step"])
        pairs = hints._pairs(x.numpy(), y.numpy(), EOS_TOKEN_ID)
        for n in hints.LENGTHS:
            table = hints.Table.from_sorted(pairs[n]).filtered(keys[n])
            kept[n].append(table)
        done += len(x)
        if done >= milestones[0]:
            merged = {n: hints.Table.merge(kept[n]) for n in hints.LENGTHS}
            kept = {n: [merged[n]] for n in hints.LENGTHS}
            path = output / f"corpus-{done}.npz"
            hints.save_tables(path, merged, rows=done)
            saved.append({"rows": done, "tokens": done * settings["context"], "path": path.name})
            log(json.dumps({**saved[-1], "elapsed_seconds": time.time() - started}))
            milestones.pop(0)
        elif done % (64 * settings["rows_per_step"]) == 0:
            # Bound memory: fold the per-batch tables together now and then.
            kept = {n: [hints.Table.merge(kept[n])] for n in hints.LENGTHS}
    result = {"saved": saved, "rows_read": done, "unreached_milestones": milestones}
    (output / "corpus.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_corpora(manifest, arm, tables, settings):
    """Validation loss of a saved hinted arm with each prefix table in `tables` (paths)."""
    import torch
    from training_data.subword import EOS_TOKEN_ID

    saved = torch.load(Path(arm) / "model.pt", map_location="cuda")
    lm = build_model(settings, saved["hinted"])
    lm.load_state_dict({k: v.float() for k, v in saved["model"].items()})
    rows = validation_rows(manifest, settings)
    own_path = Path(arm) / "training-rows-validation-tables.npz"
    own = hints.load_tables(own_path) if own_path.exists() else None
    results = []
    for path in tables:
        loaded = hints.load_tables(path)
        with np.load(path) as data:
            row_count = int(data["rows"])
        entry = {
            "rows": row_count,
            "tokens": row_count * settings["context"],
            **evaluate(lm, rows, settings, loaded),
        }
        if (
            own is not None
            and Path(path).name == f"corpus-{saved['steps'] * settings['rows_per_step']}.npz"
        ):
            # The training-prefix table must reproduce the online index's hints exactly.
            entry["matches_online_index"] = all(
                hints.lookup(own, x, EOS_TOKEN_ID) == hints.lookup(loaded, x, EOS_TOKEN_ID)
                for x, _ in micro_batches(*rows, settings["micro_rows"])
            )
        results.append(entry)
    return results
