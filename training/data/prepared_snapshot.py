"""Find an immutable raw snapshot that already satisfies this data recipe."""

import json
from pathlib import Path

from .raw_corpus import staging_identity
from .subword import TOKENIZER_SHA256, digest


def find_prepared_snapshot(folder, recipe, target_tokens):
    folder = Path(folder)
    expected = staging_identity(recipe)
    weights = {s["name"]: s["weight_percent"] for s in recipe["sources"]}
    for path in sorted((folder / "snapshots").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("format") != "raw-text-v1"
            or manifest.get("staging_identity") != expected
            or manifest.get("tokenizer_sha256") != TOKENIZER_SHA256
            or manifest.get("max_passes") != 2
        ):
            continue
        sources = manifest["sources"]
        if {name: s["weight_percent"] for name, s in sources.items()} != weights:
            continue
        if any(
            s["splits"]["train"]["loss_tokens"] * 100 * s.get("max_passes", 2)
            < target_tokens * s["weight_percent"]
            or s["splits"]["validation"]["loss_tokens"] <= 0
            for s in sources.values()
        ):
            continue
        database = manifest["database"]
        if Path(database["path"]).name != database["path"]:
            raise ValueError("Unsafe prepared database path")
        if digest(path.parent / database["path"]) != database["sha256"]:
            raise ValueError("Prepared snapshot checksum mismatch; refusing reuse")
        return path
    return None
