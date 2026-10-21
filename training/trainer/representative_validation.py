"""Validate and load a frozen, stratified held-out set from the snapshot manifest."""

import gzip
import json
from pathlib import Path

from training_data.subword import TOKENIZER_SHA256, digest


def load_heldout(manifest_path, fallback):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    artifact = manifest.get("heldout")
    if artifact is None:
        return [(name, next(fallback.blocks(name))) for name in fallback.sources]
    if artifact["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Held-out tokenizer mismatch")
    if Path(artifact["path"]).name != artifact["path"]:
        raise ValueError("Unsafe held-out path")
    source = path.parent / artifact["path"]
    if digest(source) != artifact["sha256"]:
        raise ValueError("Held-out checksum mismatch")
    blocks = []
    sources = set()
    with gzip.open(source, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            tokens, name = row["tokens"], row["source"]
            if name not in manifest["sources"] or not 2 <= len(tokens) <= 8193:
                raise ValueError("Invalid held-out window")
            if any(type(t) is not int or not 0 <= t < 32768 for t in tokens):
                raise ValueError("Invalid held-out token")
            sources.add(name)
            blocks.append(
                (
                    name,
                    {
                        "input_ids": tokens[:-1],
                        "labels": tokens[1:],
                        "loss_tokens": len(tokens) - 1,
                        "stratum": row.get("stratum", name),
                    },
                )
            )
    if sources != set(manifest["sources"]) or len(blocks) != artifact["windows"]:
        raise ValueError("Held-out coverage mismatch")
    return blocks
