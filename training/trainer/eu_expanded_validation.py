"""Family-balanced, paired held-out evaluation of EU domain continuation."""

import gzip
import hashlib
import json
import math
import random
import sqlite3
import time
import zlib
from collections import defaultdict
from pathlib import Path

from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen

from trainer.loop import atomic_json

SEED = 20261019
CONTEXT = 2048
PARENTS_PER_SOURCE = 256
WINDOWS_PER_DOCUMENT = 2


def rank_key(*parts):
    return hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()


def select_families(rows, category, limit=PARENTS_PER_SOURCE):
    """Uniform deterministic family selection, then one document per family."""
    families = {}
    missing = 0
    for key, parent, count in rows:
        if count <= 0:
            continue
        if not parent:
            missing += 1
            parent = "missing-parent/" + key
        previous = families.get(parent)
        if previous is None or rank_key(category, key) < rank_key(category, previous[0]):
            families[parent] = (key, count)
    selected = sorted(families, key=lambda p: rank_key(category, p))[:limit]
    return [(families[p][0], p, families[p][1]) for p in selected], len(families), missing


def offsets_for(key, count, context=CONTEXT, windows=WINDOWS_PER_DOCUMENT):
    """Training-style nonoverlapping windows, sampled across the whole document."""
    blocks = math.ceil(count / context)
    rng = random.Random(int(rank_key("offset", key), 16))
    return sorted(i * context for i in rng.sample(range(blocks), min(windows, blocks)))


def build_sample(original_manifest, eu_manifest, folder, commit=lambda: None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [Path(original_manifest), Path(eu_manifest)]
    identity = {
        "manifest_sha256": [digest(p) for p in paths],
        "tokenizer_sha256": TOKENIZER_SHA256,
        "seed": SEED,
        "context": CONTEXT,
        "parents_per_source": PARENTS_PER_SOURCE,
        "windows_per_document": WINDOWS_PER_DOCUMENT,
        "implementation_sha256": digest(__file__),
    }
    summary_path = folder / "sample-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            summary["identity"] != identity
            or digest(folder / "windows.jsonl.gz") != summary["windows_sha256"]
        ):
            raise ValueError("Cached evaluation sample identity mismatch")
        return summary
    tokenizer = load_frozen()
    coverage = {}
    temporary = folder / "windows.partial.gz"
    with gzip.open(temporary, "wt") as output:
        for manifest_path in paths:
            manifest = json.loads(manifest_path.read_text())
            if manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
                raise ValueError("Tokenizer mismatch")
            database = manifest_path.parent / manifest["database"]["path"]
            if digest(database) != manifest["database"]["sha256"]:
                raise ValueError("Dataset checksum mismatch")
            db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
            if manifest_path == paths[0]:
                categories = [
                    ("original/" + name, "source=?", name) for name in manifest["sources"]
                ]
            else:
                categories = [
                    (name, "json_extract(provenance,'$.institution')=?", name)
                    for name in ("ecb", "ema")
                ]
            for category, clause, value in categories:
                rows = db.execute(
                    "SELECT hash,parent,loss_tokens FROM documents "
                    "WHERE split='validation' AND conflict=0 AND " + clause,
                    (value,),
                ).fetchall()
                selected, available, missing = select_families(rows, category)
                stats = {
                    "available_documents": len(rows),
                    "available_families": available,
                    "selected_documents": len(selected),
                    "selected_families": len(selected),
                    "missing_parent_documents": missing,
                    "windows": 0,
                    "tokens": 0,
                    "nonprefix_windows": 0,
                }
                for key, parent, expected in selected:
                    raw = db.execute("SELECT text FROM documents WHERE hash=?", (key,)).fetchone()[
                        0
                    ]
                    content = tokenizer.encode(
                        zlib.decompress(raw).decode(), add_special_tokens=False
                    ).ids
                    if len(content) + 1 != expected:
                        raise ValueError("Stored document token count mismatch")
                    tokens = [BOS_TOKEN_ID, *content, EOS_TOKEN_ID]
                    parent_id = rank_key(category, "parent", parent)
                    for offset in offsets_for(key, expected):
                        y = tokens[offset + 1 : offset + CONTEXT + 1]
                        x = tokens[offset : offset + len(y)]
                        row = {
                            "window_id": rank_key(category, key, offset),
                            "category": category,
                            "parent_id": parent_id,
                            "document_hash": key,
                            "offset": offset,
                            "input_ids": x,
                            "labels": y,
                        }
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        stats["windows"] += 1
                        stats["tokens"] += len(y)
                        stats["nonprefix_windows"] += int(offset > 0)
                if not stats["tokens"]:
                    raise ValueError("No held-out data for " + category)
                coverage[category] = stats
                print(json.dumps({"category": category, **stats}), flush=True)
            db.close()
    temporary.replace(folder / "windows.jsonl.gz")
    summary = {
        "identity": identity,
        "windows_sha256": digest(folder / "windows.jsonl.gz"),
        "coverage": coverage,
        "total_tokens": sum(s["tokens"] for s in coverage.values()),
        "total_windows": sum(s["windows"] for s in coverage.values()),
        "total_families": sum(s["selected_families"] for s in coverage.values()),
        "sampling_note": "Uniform document-family sample, one document per family, "
        "up to two uniform nonoverlapping 2K blocks per document. Token-weighted "
        "loss on this sample is not a whole-corpus token-uniform estimate.",
    }
    atomic_json(summary_path, summary)
    commit()
    return summary


def score_checkpoint(checkpoint, sample_folder, output, commit=lambda: None):
    import torch
    from model_100m.model import GPT, ModelConfig

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, sample_folder, output = Path(checkpoint), Path(sample_folder), Path(output)
    summary = json.loads((sample_folder / "sample-summary.json").read_text())
    if digest(sample_folder / "windows.jsonl.gz") != summary["windows_sha256"]:
        raise ValueError("Evaluation sample checksum mismatch")
    identity = {
        "checkpoint_sha256": digest(checkpoint),
        "sample_sha256": summary["windows_sha256"],
        "implementation_sha256": digest(__file__),
        "precision": "FP32; TF32 disabled",
    }
    if output.exists():
        result = json.loads(output.read_text())
        if result["identity"] != identity:
            raise ValueError("Existing score identity mismatch")
        if result.get("completed"):
            return result
    else:
        result = {
            "identity": identity,
            "checkpoint": str(checkpoint),
            "completed": False,
            "rows": [],
        }
    done = {r["window_id"] for r in result["rows"]}
    started = time.monotonic()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Checkpoint tokenizer mismatch")
    model = (
        GPT(
            ModelConfig(
                vocab_size=32768,
                block_size=config["context"],
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_embd=config["n_embd"],
            )
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint")
    del saved
    with gzip.open(sample_folder / "windows.jsonl.gz", "rt") as handle:
        windows = [json.loads(line) for line in handle]
    pending = sorted(
        (r for r in windows if r["window_id"] not in done), key=lambda r: len(r["labels"])
    )
    with torch.inference_mode():
        for start in range(0, len(pending), 4):
            batch = pending[start : start + 4]
            length = max(len(r["labels"]) for r in batch)
            x = torch.tensor(
                [r["input_ids"] + [0] * (length - len(r["labels"])) for r in batch], device="cuda"
            )
            y = torch.tensor(
                [r["labels"] + [-100] * (length - len(r["labels"])) for r in batch], device="cuda"
            )
            logits = model(x)
            losses = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), y.flatten(), reduction="none", ignore_index=-100
            )
            sums = losses.reshape(len(batch), length).double().sum(1).cpu().tolist()
            if any(not math.isfinite(value) for value in sums):
                raise ValueError("Nonfinite validation loss")
            for row, nll in zip(batch, sums):
                result["rows"].append(
                    {k: row[k] for k in ("window_id", "category", "parent_id")}
                    | {"tokens": len(row["labels"]), "nll_sum": nll}
                )
            if start % 256 == 0:
                atomic_json(output, result)
                commit()
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint.parent.name,
                            "windows_scored": len(result["rows"]),
                            "total_windows": len(windows),
                        }
                    ),
                    flush=True,
                )
    if {r["window_id"] for r in result["rows"]} != {r["window_id"] for r in windows}:
        raise ValueError("Incomplete matched scoring")
    result.update(
        completed=True,
        elapsed_seconds=time.monotonic() - started,
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
        total_tokens=sum(r["tokens"] for r in result["rows"]),
    )
    atomic_json(output, result)
    commit()
    return result


def paired_summary(before, after, sample, bootstrap_samples=2000):
    import numpy as np

    left = {r["window_id"]: r for r in before["rows"]}
    right = {r["window_id"]: r for r in after["rows"]}
    if (
        left.keys() != right.keys()
        or before["identity"]["sample_sha256"] != after["identity"]["sample_sha256"]
    ):
        raise ValueError("Checkpoints did not score identical windows")
    grouped = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    for key, row in left.items():
        other = right[key]
        if any(row[k] != other[k] for k in ("tokens", "parent_id", "category")):
            raise ValueError("Matched window metadata differs")
        values = grouped[row["category"]][row["parent_id"]]
        values[0] += row["tokens"]
        values[1] += row["nll_sum"]
        values[2] += other["nll_sum"]
    categories = {}
    for name, parents in sorted(grouped.items()):
        values = np.array(list(parents.values()), dtype=np.float64)
        tokens, first, second = values.sum(axis=0)
        rng = np.random.default_rng(int(rank_key("bootstrap", name)[:16], 16))
        draws = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        sums = values[draws].sum(axis=1)
        deltas = (sums[:, 2] - sums[:, 1]) / sums[:, 0]
        low, high = np.quantile(deltas, [0.025, 0.975])
        categories[name] = {
            "before_loss": first / tokens,
            "after_loss": second / tokens,
            "delta_loss": (second - first) / tokens,
            "delta_ci95": [float(low), float(high)],
            "perplexity_ratio_after_before": math.exp((second - first) / tokens),
            "tokens": int(tokens),
            "families": len(parents),
            "direction": "worse" if low > 0 else "better" if high < 0 else "uncertain",
        }
    return {
        "categories": categories,
        "sample": sample,
        "before_identity": before["identity"],
        "after_identity": after["identity"],
        "bootstrap_samples": bootstrap_samples,
        "limitations": [
            "Validation split only; no test-set tuning or scoring.",
            "Family-balanced sample, not corpus-token-uniform perplexity.",
            "CIs are conditional on selected families and are not simultaneous across categories.",
            "Source split/near-duplicate/contamination audits remain pending.",
            "2K window loss does not establish 8K long-context quality or generation correctness.",
        ],
    }
