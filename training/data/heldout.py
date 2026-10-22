"""Frozen held-out windows that a snapshot manifest references under `heldout`.

A snapshot can ship a fixed, stratified evaluation set instead of letting the trainer
sample its first validation windows. The file is gzip JSON lines, one window per line:
`{"source": ..., "stratum": ..., "tokens": [BOS, ..., EOS?]}` plus any provenance
fields. The manifest entry pins its file name, checksum, tokenizer and window count.
"""

import gzip
import json
from pathlib import Path

from .subword import TOKENIZER_SHA256, VOCAB_SIZE, digest


def write_heldout(folder, rows, name="heldout.jsonl.gz"):
    """Write windows next to a manifest and return its `heldout` entry."""
    path = Path(folder) / name
    if path.exists():
        raise ValueError("Refuse to overwrite a frozen held-out set")
    windows = 0
    with gzip.open(path, "wt") as handle:
        for row in rows:
            if not isinstance(row.get("source"), str) or len(row["tokens"]) < 2:
                raise ValueError("Held-out windows need a source and at least two tokens")
            handle.write(json.dumps(row) + "\n")
            windows += 1
    return {
        "path": name,
        "sha256": digest(path),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "windows": windows,
    }


def load_heldout(manifest_path, context):
    """Validated windows grouped by source, as evaluation blocks with a `stratum`."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest["heldout"]
    if artifact["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Held-out tokenizer mismatch")
    if Path(artifact["path"]).name != artifact["path"]:
        raise ValueError("Unsafe held-out path")
    path = manifest_path.parent / artifact["path"]
    if digest(path) != artifact["sha256"]:
        raise ValueError("Held-out checksum mismatch")
    grouped = {name: [] for name in manifest["sources"]}
    windows = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            tokens, name = row["tokens"], row["source"]
            if name not in grouped or not 2 <= len(tokens) <= context + 1:
                raise ValueError("Invalid held-out window")
            if any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in tokens):
                raise ValueError("Invalid held-out token")
            grouped[name].append(
                {
                    "input_ids": tokens[:-1],
                    "labels": tokens[1:],
                    "loss_tokens": len(tokens) - 1,
                    "stratum": row.get("stratum", name),
                }
            )
            windows += 1
    if windows != artifact["windows"] or not all(grouped.values()):
        raise ValueError("Held-out set must match its window count and cover every source")
    return grouped
