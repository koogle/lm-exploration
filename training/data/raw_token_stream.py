"""Tokenize selected raw documents only as needed to fill training windows."""

import json
import sqlite3
import zlib
from pathlib import Path

from .subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    FIRST_CONTENT_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    PAD_TOKEN_ID,
    TOKENIZER_SHA256,
    digest,
    load_frozen,
)
from .token_stream import TokenMixture


class RawTokenMixture(TokenMixture):
    """The same token-deficit scheduler, with resumable raw-document cursors."""

    def __init__(self, manifest_path, context=2048, split="train", allow_unreviewed=False):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("format") not in ("raw-text-v1", "scholarly-raw-v1"):
            raise ValueError("Unexpected raw manifest format")
        if context < 1 or split not in ("train", "validation"):
            raise ValueError("Invalid context or split")
        if not self.manifest["training_ready"] and not allow_unreviewed:
            raise ValueError("Dataset has pending audits; allow_unreviewed_data required")
        if (
            self.manifest["tokenizer_sha256"] != TOKENIZER_SHA256
            or self.manifest["max_passes"] != 2
        ):
            raise ValueError("Tokenizer or repetition policy mismatch")
        artifact = self.manifest["database"]
        if Path(artifact["path"]).name != artifact["path"]:
            raise ValueError("Unsafe database path")
        database = self.path.parent / artifact["path"]
        if digest(database) != artifact["sha256"]:
            raise ValueError("Raw database checksum mismatch")
        self.db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
        self.context, self.split = context, split
        self.identity = {
            "manifest_sha256": digest(self.path),
            "context": context,
            "split": split,
            "tokenizer_sha256": TOKENIZER_SHA256,
        }
        self.sources = self.manifest["sources"]
        if any(s["splits"][split]["loss_tokens"] <= 0 for s in self.sources.values()):
            raise ValueError("Every mixture source needs data for this split")
        self.positions = dict.fromkeys(self.sources, 0)
        self.exposures = dict.fromkeys(self.sources, 0)
        self.iterators = {}
        self.cursors = {name: {"pass": 0, "hash": "", "offset": 0} for name in self.sources}
        self.tokenizer = load_frozen()
        self.single_pass = self.manifest.get("continuation_passes") == 1

    def blocks(self, name):
        saved = dict(self.cursors[name])
        passes = 2 if self.split == "train" and not self.single_pass else 1
        for pass_index in range(saved["pass"], passes):
            start_hash = saved["hash"] if pass_index == saved["pass"] else ""
            rows = self.db.execute(
                """SELECT hash,text,loss_tokens FROM documents
                WHERE source=? AND split=? AND conflict=0 AND hash>=? ORDER BY hash""",
                (name, self.split, start_hash),
            )
            for key, raw, expected_count in rows:
                ids = self.tokenizer.encode(
                    zlib.decompress(raw).decode(), add_special_tokens=False
                ).ids
                if len(ids) + 1 != expected_count or any(t < FIRST_CONTENT_TOKEN_ID for t in ids):
                    raise ValueError("Raw document/token count mismatch")
                tokens = [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID]
                start = (
                    saved["offset"] if pass_index == saved["pass"] and key == saved["hash"] else 0
                )
                for offset in range(start, len(tokens) - 1, self.context):
                    y = tokens[offset + 1 : offset + self.context + 1]
                    count = len(y)
                    x = tokens[offset : offset + count]
                    self.cursors[name] = {
                        "pass": pass_index,
                        "hash": key,
                        "offset": offset + self.context,
                    }
                    yield {
                        "input_ids": x + [PAD_TOKEN_ID] * (self.context - count),
                        "labels": y + [LOSS_IGNORE_INDEX] * (self.context - count),
                        "loss_tokens": count,
                        "source": name,
                        "pass": pass_index + 1,
                    }

    def next_block(self):
        name = min(
            self.sources, key=lambda n: (self.exposures[n] / self.sources[n]["weight_percent"], n)
        )
        if name not in self.iterators:
            self.iterators[name] = self.blocks(name)
        block = next(self.iterators[name])
        self.positions[name] += 1
        self.exposures[name] += block["loss_tokens"]
        return block

    def next_batch(self, batch_size, device="cpu"):
        if not self.single_pass:
            return super().next_batch(batch_size, device)
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state = self.state_dict()
        blocks = []
        try:
            for _ in range(batch_size):
                try:
                    blocks.append(self.next_block())
                except StopIteration:
                    break
            if not blocks:
                raise StopIteration("Completed the single continuation pass")
            # Train every final target exactly once; empty windows still join DDP.
            blocks.extend(
                {
                    "input_ids": [PAD_TOKEN_ID] * self.context,
                    "labels": [LOSS_IGNORE_INDEX] * self.context,
                }
                for _ in range(batch_size - len(blocks))
            )
            return (
                torch.tensor([b["input_ids"] for b in blocks], dtype=torch.long, device=device),
                torch.tensor([b["labels"] for b in blocks], dtype=torch.long, device=device),
            )
        except Exception:
            self.load_state_dict(state)
            raise

    def state_dict(self):
        # Raw-document cursors are not the seekable token-shard cursor schema.
        return {
            **self.identity,
            "positions": dict(self.positions),
            "loss_token_exposures": dict(self.exposures),
            "raw_cursors": {k: dict(v) for k, v in self.cursors.items()},
        }

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Checkpoint data identity mismatch")
        for key in ("positions", "loss_token_exposures"):
            if set(state[key]) != set(self.sources) or any(
                type(v) is not int or v < 0 for v in state[key].values()
            ):
                raise ValueError("Invalid raw data cursor")
        cursors = state["raw_cursors"]
        if set(cursors) != set(self.sources):
            raise ValueError("Raw source cursor mismatch")
        for cursor in cursors.values():
            if (
                type(cursor["pass"]) is not int
                or cursor["pass"] not in (0, 1)
                or type(cursor["offset"]) is not int
                or cursor["offset"] < 0
                or cursor["offset"] % self.context
                or not isinstance(cursor["hash"], str)
            ):
                raise ValueError("Invalid raw cursor")
        for iterator in self.iterators.values():
            iterator.close()
        self.iterators.clear()
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.cursors = {k: dict(v) for k, v in cursors.items()}

    def close(self):
        super().close()
        self.db.close()
