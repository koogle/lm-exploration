"""Pretraining text replayed during fine-tuning, drawn from a raw snapshot.

Each window is `context + 1` tokens of consecutive `BOS ids EOS` documents of one source,
trained with the ordinary pretraining loss (every target except BOS). Sources are chosen
by the given weights, which need not match the snapshot's mixture; a window starts at a
uniformly random stored row of its source's training split. Validation windows are the
first packed windows of each source's validation split, so they are fixed.
"""

import json
import random
import sqlite3
import zlib
from pathlib import Path

from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    digest,
)

# Stored rows read per query while filling a window.
ROWS_PER_READ = 32


class Replay:
    def __init__(self, snapshot, weights, context, tokenizer, seed):
        snapshot = Path(snapshot)
        self.manifest_path = snapshot / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text())
        if manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Replay snapshot uses another tokenizer")
        if set(weights) - set(manifest["sources"]) or not all(w > 0 for w in weights.values()):
            raise ValueError("Replay weights must name snapshot sources with positive weights")
        database = snapshot / manifest["database"]["path"]
        self.db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
        self.sources = sorted(weights)
        self.weights = [weights[s] for s in self.sources]
        self.context, self.tokenizer = context, tokenizer
        self.rng = random.Random(seed)
        self.bounds = {}
        for source in self.sources:
            low, high = self.db.execute(
                """SELECT MIN(rowid), MAX(rowid) FROM documents
                WHERE source=? AND split='train' AND conflict=0""",
                (source,),
            ).fetchone()
            if low is None:
                raise ValueError(f"Replay source {source} has no training documents")
            self.bounds[source] = (low, high)

    def identity(self):
        return {"manifest_sha256": digest(self.manifest_path), "sources": self.sources}

    def tokens(self, source, split, start, count):
        """`count` tokens of consecutive documents of `source` from row `start` on."""
        tokens, row = [], start
        while len(tokens) < count:
            rows = self.db.execute(
                """SELECT rowid, text FROM documents WHERE source=? AND split=? AND conflict=0
                AND rowid>=? ORDER BY rowid LIMIT ?""",
                (source, split, row, ROWS_PER_READ),
            ).fetchall()
            if not rows:
                if split != "train" or row == self.bounds[source][0]:
                    raise ValueError(f"Not enough {split} text in {source}")
                row = self.bounds[source][0]  # wrap to the source's first training row
                continue
            texts = [zlib.decompress(blob).decode() for _, blob in rows]
            for encoded in self.tokenizer.encode_batch(texts, add_special_tokens=False):
                tokens += [BOS_TOKEN_ID, *encoded.ids, EOS_TOKEN_ID]
            row = rows[-1][0] + 1
        return tokens[:count]

    def example(self, tokens, source):
        labels = [t if t != BOS_TOKEN_ID else LOSS_IGNORE_INDEX for t in tokens[1:]]
        return {"bucket": source, "x": tokens[:-1], "y": labels}

    def sample(self):
        """One training window from a source chosen by weight."""
        source = self.rng.choices(self.sources, self.weights)[0]
        start = self.rng.randint(*self.bounds[source])
        return self.example(self.tokens(source, "train", start, self.context + 1), source)

    def validation(self, count):
        """The first `count` validation windows of each source."""
        windows = []
        for source in self.sources:
            tokens = self.tokens(source, "validation", 0, count * self.context + 1)
            for index in range(count):
                start = index * self.context
                windows.append(self.example(tokens[start : start + self.context + 1], source))
        return windows
