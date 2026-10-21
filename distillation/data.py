"""Read existing immutable code databases; preserve splits and parent exclusions."""

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path


class CodeWindows:
    def __init__(
        self,
        manifest,
        database,
        tokenizer,
        context=256,
        split="train",
        state=None,
        root="/data",
    ):
        self.manifest = json.loads(Path(manifest).read_text())
        self.identity = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        self.tokenizer, self.context, self.split = tokenizer, context, split
        if split not in ("train", "validation") or context < 2:
            raise ValueError("Invalid window settings")
        database = Path(database)
        matches = [
            x
            for x in self.manifest["external_databases"]
            if x["path"].endswith("/" + str(database.relative_to(root)))
        ]
        if len(matches) != 1 or database.stat().st_size != matches[0]["size_bytes"]:
            raise ValueError("Database does not match the snapshot inventory")
        if matches[0]["sources"] != {"code": "code"}:
            raise ValueError("Expected code-only database")
        self.db = sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)
        self.excluded_hashes = set(self.manifest["excluded_validation_hashes"])
        self.excluded_parents = set(self.manifest["excluded_validation_parents"])
        self.state = dict(state or {"hash": "", "offset": 0})
        self.iterator = self._windows()

    def _windows(self):
        rows = self.db.execute(
            """SELECT hash,parent,text,loss_tokens FROM documents
            WHERE source='code' AND split=? AND conflict=0 AND hash>=? ORDER BY hash""",
            (self.split, self.state["hash"]),
        )
        for key, parent, raw, expected in rows:
            if self.split == "train" and (
                key in self.excluded_hashes or parent in self.excluded_parents
            ):
                continue
            text = zlib.decompress(raw).decode("utf-8")
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) + 1 != expected or any(t < 4 for t in ids):
                raise ValueError(
                    "Frozen tokenizer does not reproduce corpus token counts"
                )
            tokens = [1, *ids, 2]
            start = self.state["offset"] if key == self.state["hash"] else 0
            for offset in range(start, len(tokens) - 1, self.context):
                x = tokens[offset : offset + self.context]
                y = tokens[offset + 1 : offset + self.context + 1]
                x = x[: len(y)]
                self.state = {"hash": key, "offset": offset + len(y)}
                yield x, y

    def next(self):
        return next(self.iterator)
