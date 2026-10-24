"""Shared constants and small helpers for the distillation validation ladder."""

import hashlib
import json
import os
from pathlib import Path

DATASET = "jfrick/qwen35-python-math-trajectories-20261022"
CAMPAIGN = "qwen35-python-math-8h-20261022-01"
TEACHER = "Qwen/Qwen3.5-9B"
TEACHER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MBPP = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

QWEN_VOCAB = 248320
QWEN_ENDOFTEXT = 248044
TOPK = 16  # teacher entries kept per position; the rest is one residual bucket
HELDOUT_PERCENT = 2  # chunks whose path hash falls below this go to held-out
SPLIT_SALT = "distillation-validation-v1"


def path_hash(path, salt=SPLIT_SALT):
    return int.from_bytes(hashlib.sha256(f"{salt}:{path}".encode()).digest()[:8], "big")


def is_heldout(path):
    return path_hash(path) % 100 < HELDOUT_PERCENT


def training_order(paths):
    """Fixed random chunk order; every token budget is a prefix of this order."""
    return sorted(paths, key=lambda p: path_hash(p, SPLIT_SALT + ":order"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
