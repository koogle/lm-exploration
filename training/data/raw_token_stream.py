"""Tokenize selected raw documents only as needed to fill packed training windows.

Each source's documents are concatenated, as `BOS ids EOS`, into full windows, so
short documents do not waste a window on padding. BOS is never a target; every
content token and EOS is a target exactly once per pass.

Each source is read at most `max_passes` times (1 or 2; default 2). A source may also
set an exact `target_tokens` budget: the scheduler stops drawing from it once the budget
is reached, masking any excess targets in the last window. When every budget is met, or
a source runs out of passes, the final partial batch is padded rather than dropped.
"""

import json
import math
import sqlite3
import zlib
from fractions import Fraction
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
        if self.manifest.get("format") != "raw-text-v1":
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
        if "external_databases" in self.manifest:
            from .prepared_reference import attach_prepared

            attach_prepared(self.db, self.manifest)
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
        self.passes = {}
        for name, source in self.sources.items():
            cap = source.get("max_passes", 2)
            if type(cap) is not int or cap not in (1, 2):
                raise ValueError("Source pass cap must be one or two")
            self.passes[name] = cap if split == "train" else 1
        self.budgets = {}
        if split == "train" and any("target_tokens" in s for s in self.sources.values()):
            for name, source in self.sources.items():
                budget = source.get("target_tokens")
                capacity = source["splits"]["train"]["loss_tokens"] * self.passes[name]
                if type(budget) is not int or not 0 < budget <= capacity:
                    raise ValueError(f"Source {name} needs a target within its pass capacity")
                self.budgets[name] = budget
        self.positions = dict.fromkeys(self.sources, 0)
        self.exposures = dict.fromkeys(self.sources, 0)
        self.iterators = {}
        self.cursors = {name: {"pass": 0, "hash": "", "offset": 0} for name in self.sources}
        self.tokenizer = load_frozen()

    def blocks(self, name):
        saved = dict(self.cursors[name])
        for pass_index in range(saved["pass"], self.passes[name]):
            resume = pass_index == saved["pass"]
            rows = self.db.execute(
                """SELECT hash,text,loss_tokens FROM documents
                WHERE source=? AND split=? AND conflict=0 AND hash>=? ORDER BY hash""",
                (name, self.split, saved["hash"] if resume else ""),
            )
            buffer = []
            # [document hash, offset of the buffer's first token in it, tokens left]
            segments = []
            for key, raw, expected_count in rows:
                ids = self.tokenizer.encode(
                    zlib.decompress(raw).decode(), add_special_tokens=False
                ).ids
                if len(ids) + 1 != expected_count or any(t < FIRST_CONTENT_TOKEN_ID for t in ids):
                    raise ValueError("Raw document/token count mismatch")
                tokens = [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID]
                start = saved["offset"] if resume and key == saved["hash"] else 0
                if start >= len(tokens):
                    continue
                buffer.extend(tokens[start:])
                segments.append([key, start, len(tokens) - start])
                while len(buffer) > self.context:
                    window = buffer[: self.context + 1]
                    # The next window starts at this window's last target.
                    del buffer[: self.context]
                    consume(segments, self.context)
                    self.cursors[name] = {
                        "pass": pass_index,
                        "hash": segments[0][0],
                        "offset": segments[0][1],
                    }
                    yield self.window(name, pass_index, window)
            if len(buffer) > 1:
                key, start, length = segments[-1]
                self.cursors[name] = {"pass": pass_index, "hash": key, "offset": start + length}
                yield self.window(name, pass_index, buffer)

    def window(self, name, pass_index, tokens):
        x, y = tokens[:-1], tokens[1:]
        labels = [LOSS_IGNORE_INDEX if t == BOS_TOKEN_ID else t for t in y]
        padding = self.context - len(x)
        return {
            "input_ids": x + [PAD_TOKEN_ID] * padding,
            "labels": labels + [LOSS_IGNORE_INDEX] * padding,
            "loss_tokens": sum(t != LOSS_IGNORE_INDEX for t in labels),
            "source": name,
            "pass": pass_index + 1,
        }

    def next_block(self):
        eligible = [
            n for n in self.sources if not self.budgets or self.exposures[n] < self.budgets[n]
        ]
        if not eligible:
            raise StopIteration("Every source target is complete")
        name = min(
            eligible, key=lambda n: (self.exposures[n] / self.sources[n]["weight_percent"], n)
        )
        if name not in self.iterators:
            self.iterators[name] = self.blocks(name)
        block = next(self.iterators[name])
        if self.budgets:
            remaining = self.budgets[name] - self.exposures[name]
            if block["loss_tokens"] > remaining:
                kept = 0
                for index, label in enumerate(block["labels"]):
                    if label != LOSS_IGNORE_INDEX:
                        if kept == remaining:
                            block["labels"][index] = LOSS_IGNORE_INDEX
                        else:
                            kept += 1
                block["loss_tokens"] = remaining
        self.positions[name] += 1
        self.exposures[name] += block["loss_tokens"]
        return block

    def next_batch(self, batch_size, device="cpu"):
        """Like TokenMixture.next_batch, but pads the final partial batch.

        Every remaining target is trained once; padded windows still join DDP with no loss.
        """
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
                raise StopIteration("Raw mixture exhausted")
            padding = {
                "input_ids": [PAD_TOKEN_ID] * self.context,
                "labels": [LOSS_IGNORE_INDEX] * self.context,
            }
            blocks += [padding] * (batch_size - len(blocks))
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
        for name, cursor in cursors.items():
            if (
                type(cursor["pass"]) is not int
                or not 0 <= cursor["pass"] < self.passes[name]
                or type(cursor["offset"]) is not int
                or cursor["offset"] < 0
                or not isinstance(cursor["hash"], str)
            ):
                raise ValueError("Invalid raw cursor")
        if self.budgets and any(
            state["loss_token_exposures"][name] > budget for name, budget in self.budgets.items()
        ):
            raise ValueError("Restored exposure exceeds a source target")
        for iterator in self.iterators.values():
            iterator.close()
        self.iterators.clear()
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.cursors = {k: dict(v) for k, v in cursors.items()}

    def close(self):
        super().close()
        self.db.close()


def consume(segments, count):
    """Drop `count` tokens from the front of the buffer's document segments."""
    while count:
        segment = segments[0]
        taken = min(count, segment[2])
        segment[1] += taken
        segment[2] -= taken
        count -= taken
        if not segment[2]:
            segments.pop(0)


def token_targets(weights, total):
    """Exact integer per-source targets summing to `total` (largest-remainder rounding).

    `weights` maps source names to percentages summing to 100; ties go to the name that
    sorts first, so the allocation is deterministic.
    """
    weights = {name: Fraction(str(share)) for name, share in weights.items()}
    if type(total) is not int or total < 1 or sum(weights.values()) != 100:
        raise ValueError("Expected a positive integer total and weights summing to 100")
    exact = {name: total * share / 100 for name, share in weights.items()}
    targets = {name: math.floor(value) for name, value in exact.items()}
    left = total - sum(targets.values())
    for name in sorted(exact, key=lambda n: (-(exact[n] - targets[n]), n))[:left]:
        targets[name] += 1
    return targets
