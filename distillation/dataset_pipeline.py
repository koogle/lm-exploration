"""Append-only reusable dataset, independent of the teacher pool's size/speed."""

import hashlib
import json
import math
from pathlib import Path

from distillation.scale_data import atomic_json, available_chunks


def teacher_count(student_tps, teacher_tps, headroom=1.25):
    if student_tps <= 0 or teacher_tps <= 0 or headroom < 1:
        raise ValueError("Positive rates and headroom >= 1 required")
    return max(1, math.ceil(student_tps * headroom / teacher_tps))


def publish_chunks(root, workers):
    """Append all newly committed shards; a slow worker cannot block its peers."""
    root = Path(root)
    path = root / "manifest.json"
    manifest = (
        json.loads(path.read_text())
        if path.exists()
        else {"format": "append-only-qwen-v1", "chunks": [], "status": "generating"}
    )
    seen = {(w, i): info for w, i, info in manifest["chunks"]}
    for worker in range(workers):
        state_path = root / f"producer-{worker}" / "progress.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        for index, chunk in enumerate(state["chunks"]):
            if (worker, index) in seen:
                if seen[(worker, index)] != chunk:
                    raise ValueError("Committed dataset shard changed")
            else:
                manifest["chunks"].append([worker, index, chunk])
    manifest["qwen_tokens"] = sum(c[2]["qwen_tokens"] for c in manifest["chunks"])
    manifest["records"] = sum(c[2]["records"] for c in manifest["chunks"])
    atomic_json(path, manifest)
    return manifest


def freeze_initial_prefix(root, name, continuation):
    root = Path(root)
    path = root / name / "dataset-plan.json"
    if path.exists():
        return json.loads(path.read_text())
    plan = {
        "format": "fresh-single-pass-v2",
        "initial_corpus": root.name,
        "initial_chunks": available_chunks(root),
        "continuation_corpus": continuation,
        "policy": "Immutable initial prefix followed by append-only fresh shards; no replay",
    }
    atomic_json(path, plan)
    return plan


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
