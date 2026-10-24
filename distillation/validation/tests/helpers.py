"""Shared test setup: import paths and small synthetic fixtures."""

import gzip
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "training")]
sys.modules.setdefault("model_10m", importlib.import_module("10m"))
sys.modules.setdefault("model_10m.model", importlib.import_module("10m.model"))

from common import TOPK


def chunk_file(directory, records):
    path = Path(directory) / "chunk.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def record(tokens, accepted=True, finish="stop"):
    rng = np.random.default_rng(len(tokens))
    ids = [[t] + rng.integers(0, 50, 63).tolist() for t in tokens]
    logprobs = [
        sorted(np.log(rng.dirichlet(np.ones(64)) * 0.99).tolist(), reverse=True)
    ]
    return {
        "id": f"r{tokens[0]}",
        "prompt": "p",
        "text": json.dumps(
            {"problem": "x", "code": "def solve():\n    return 1", "explanation": "e"}
        ),
        "accepted": accepted,
        "finish_reason": finish,
        "token_ids": tokens,
        "targets": {"topk_ids": ids, "topk_logprobs": logprobs * len(tokens)},
    }


def tiny_shard(directory, documents):
    rng = np.random.default_rng(1)
    tokens = np.concatenate(documents).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in documents])])
    ids = np.stack(
        [tokens] + [rng.integers(0, 40, tokens.size) for _ in range(TOPK - 1)], 1
    )
    lp = np.full((tokens.size, TOPK), math.log(0.9 / TOPK), dtype=np.float16)
    lp[:, 0] = math.log(0.1)
    shard = Path(directory)
    shard.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "tokens": tokens,
        "offsets": offsets,
        "topk_ids": ids.astype(np.int32),
        "topk_logprobs": lp,
    }.items():
        np.save(shard / f"{name}.npy", array)
    return shard
