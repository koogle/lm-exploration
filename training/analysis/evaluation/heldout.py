"""Held-out windows from a raw snapshot's validation split, for bits per byte."""

import json
import sqlite3
import zlib
from contextlib import closing

from .scoring import packed_windows

# Characters of documents to read per window; generous, since windows are cut from tokens.
CHARACTERS_PER_WINDOW = 2048 * 8


def documents(snapshot, tokenizer, count):
    """Tokenized validation documents per source, in the raw loader's hash order."""
    manifest = json.loads((snapshot / "manifest.json").read_text())
    database = snapshot / manifest["database"]["path"]
    found = {}
    with closing(
        sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    ) as db:
        for source in manifest["sources"]:
            texts = []
            for (blob,) in db.execute(
                """SELECT text FROM documents WHERE source=? AND split='validation'
                AND conflict=0 ORDER BY hash""",
                (source,),
            ):
                texts.append(zlib.decompress(blob).decode())
                if sum(map(len, texts)) > count * CHARACTERS_PER_WINDOW:
                    break
            found[source] = [e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
    return manifest, found


def windows(snapshot, model, tokenizer, count):
    """({source: windows}, {source: mixture weight}) for the first `count` windows each."""
    manifest, found = documents(snapshot, tokenizer, count)
    packed = {
        source: packed_windows(ids, model.bos_token_id, model.eos_token_id, model.block_size, count)
        for source, ids in found.items()
    }
    weights = {source: entry["weight_percent"] for source, entry in manifest["sources"].items()}
    return packed, weights
