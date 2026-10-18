"""Mix pinned upstream documents as training needs them, without a full-corpus gate.

Prefetched raw rows do not advance checkpoint cursors. The append-only seen ledger
prevents repeats and raises on newly discovered split conflicts. Its configurable
size cap prevents unbounded memory; large-scale indexed ledgers are future work.
"""

import copy
import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cached_shards import CachedShards
from .persistent_stream import PersistentStream
from .scholarly import SPLIT_VERSION, adapt
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen
from .token_stream import TokenMixture


class SplitConflict(ValueError):
    pass


class LiveTokenMixture(TokenMixture):
    def __init__(
        self, manifest_path, context=2048, split="train", allow_unreviewed=False, *, reader=None
    ):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("format") != "scholarly-live-v1" or split not in (
            "train",
            "validation",
        ):
            raise ValueError("Expected live manifest and supported split")
        if not allow_unreviewed:
            raise ValueError("Live data requires the explicit experimental-data setting")
        if (
            self.manifest.get("tokenizer_sha256") != TOKENIZER_SHA256
            or self.manifest.get("max_passes") != 2
        ):
            raise ValueError("Tokenizer or repetition policy changed")
        if self.manifest.get("split_policy") != SPLIT_VERSION:
            raise ValueError("Split policy changed")
        self.context, self.split = context, split
        self.recipe = self.manifest["recipe"]
        # Validate inline recipe using the same source-pin/weight contract.
        if sum(s["weight_percent"] for s in self.recipe["sources"]) != 100:
            raise ValueError("Source weights must sum to 100")
        self.sources = {s["name"]: s for s in self.recipe["sources"]}
        if len(self.sources) != len(self.recipe["sources"]):
            raise ValueError("Duplicate sources")
        for s in self.sources.values():
            if (
                len(s["revision"]) != 40
                or any(c not in "0123456789abcdef" for c in s["revision"])
                or not s["files"]
            ):
                raise ValueError("Every live source must pin a revision and nonempty shard list")
        self.identity = {
            "manifest_sha256": digest(self.path),
            "context": context,
            "split": split,
            "tokenizer_sha256": TOKENIZER_SHA256,
        }
        self.positions = dict.fromkeys(self.sources, 0)
        self.exposures = dict.fromkeys(self.sources, 0)
        self.iterators = {}
        self.cursors = {
            n: {"pass": 0, "shard": 0, "row": 0, "tokens": None, "offset": 0} for n in self.sources
        }
        self.events = []
        self.seen = {}
        self.maximum_seen = self.manifest.get("max_seen_documents", 5_000_000)
        self.tokenizer = load_frozen()
        self.reader = reader or CachedShards()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.streams = {n: PersistentStream(self.reader) for n in self.sources}
        self.pending = {}
        self.raw = {n: deque() for n in self.sources}
        self.started = False

    def _chunk(self, name, shard, start):
        source = self.sources[name]
        for attempt in range(4):
            rows = self.streams[name](source, source["files"][shard], start)
            result = []
            size = 0
            end = False
            try:
                while len(result) < 16 and size < 4 * 1024**2:
                    try:
                        index, row = next(rows)
                    except StopIteration:
                        end = True
                        break
                    result.append((index, row))
                    size += len(row.get("content", row.get("text", "")) or "")
                return result, end
            except Exception as exc:
                retry = isinstance(exc, (OSError, EOFError)) or type(exc).__module__.startswith(
                    ("requests", "urllib3", "httpx", "httpcore", "huggingface_hub")
                )
                if not retry or attempt == 3:
                    raise
                self.streams[name].close()
                time.sleep(2**attempt)
            finally:
                rows.close()

    def _schedule(self, name):
        c = self.cursors[name]
        if (
            name not in self.pending
            and not self.raw[name]
            and c["shard"] < len(self.sources[name]["files"])
        ):
            self.pending[name] = self.pool.submit(self._chunk, name, c["shard"], c["row"])

    def _row(self, name):
        if not self.started:
            for n in self.sources:
                self._schedule(n)
            self.started = True
        c = self.cursors[name]
        while c["shard"] < len(self.sources[name]["files"]):
            self._schedule(name)
            if not self.raw[name]:
                rows, end = self.pending.pop(name).result()
                self.raw[name].extend(rows)
                if end:
                    self.raw[name].append(None)
            item = self.raw[name].popleft()
            if item is None:
                c["shard"] += 1
                c["row"] = 0
                continue
            index, row = item
            location = (name, c["shard"], index)
            c["row"] = index + 1
            if not self.raw[name]:
                self._schedule(name)
            return row, location
        return None, None

    def _document(self, name):
        c = self.cursors[name]
        while True:
            row, location = self._row(name)
            if row is None:
                if c["pass"] + 1 >= (2 if self.split == "train" else 1):
                    raise StopIteration
                c.update({"pass": c["pass"] + 1, "shard": 0, "row": 0})
                self.streams[name].close()
                continue
            document, _reason = adapt(self.sources[name], row)
            if document is None:
                continue
            key = hashlib.sha256(document["text"].encode()).digest()
            previous = self.seen.get(key)
            if previous is not None:
                old_split, owner = previous
                if old_split != document["split"]:
                    raise SplitConflict(
                        "New cross-split duplicate encountered; pause for corpus reconciliation"
                    )
                if c["pass"] == 0 or owner != location:
                    continue
            else:
                if len(self.events) >= self.maximum_seen:
                    raise ValueError("Live seen-document ledger reached its configured memory cap")
                entry = (key, document["split"], location)
                self.events.append(entry)
                self.seen[key] = (document["split"], location)
            if document["split"] != self.split:
                continue
            ids = tuple(self.tokenizer.encode(document["text"], add_special_tokens=False).ids)
            if not ids or any(t < FIRST_CONTENT_TOKEN_ID for t in ids):
                continue
            return (1, *ids, 2)

    def blocks(self, name):
        c = self.cursors[name]
        while True:
            if c["tokens"] is None:
                try:
                    c["tokens"] = self._document(name)
                except StopIteration:
                    return
                c["offset"] = 0
            tokens = c["tokens"]
            offset = c["offset"]
            y = list(tokens[offset + 1 : offset + self.context + 1])
            count = len(y)
            x = list(tokens[offset : offset + count])
            c["offset"] += self.context
            if c["offset"] >= len(tokens) - 1:
                c["tokens"] = None
                c["offset"] = 0
            yield {
                "input_ids": x + [0] * (self.context - count),
                "labels": y + [-100] * (self.context - count),
                "loss_tokens": count,
                "source": name,
                "pass": c["pass"] + 1,
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
        import torch

        # Only small cursors are copied per batch. Full ledgers are materialized
        # at checkpoint boundaries, not copied into every buffered packet.
        before = self.cursor_state()
        try:
            blocks = [self.next_block() for _ in range(batch_size)]
            return (
                torch.tensor([b["input_ids"] for b in blocks], dtype=torch.long, device=device),
                torch.tensor([b["labels"] for b in blocks], dtype=torch.long, device=device),
            )
        except Exception:
            state = self.state_at(before)
            self.load_state_dict(state)
            raise

    def cursor_state(self):
        return {
            **self.identity,
            "positions": dict(self.positions),
            "loss_token_exposures": dict(self.exposures),
            "live_cursors": {n: dict(c) for n, c in self.cursors.items()},
            "seen_count": len(self.events),
        }

    def state_at(self, cursor):
        return {**copy.deepcopy(cursor), "seen_events": list(self.events[: cursor["seen_count"]])}

    def state_dict(self):
        return self.state_at(self.cursor_state())

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Live checkpoint identity changed")
        self.pool.shutdown(wait=True, cancel_futures=True)
        for stream in self.streams.values():
            stream.close()
        if set(state["live_cursors"]) != set(self.sources) or state["seen_count"] != len(
            state["seen_events"]
        ):
            raise ValueError("Invalid live checkpoint cursors")
        self.cursors = copy.deepcopy(state["live_cursors"])
        for c in self.cursors.values():
            if c["pass"] not in (0, 1) or min(c["shard"], c["row"], c["offset"]) < 0:
                raise ValueError("Invalid live cursor")
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.events = list(state["seen_events"])
        self.seen = {key: (split, tuple(owner)) for key, split, owner in self.events}
        self.iterators = {}
        self.pending = {}
        self.raw = {n: deque() for n in self.sources}
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.started = False

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for stream in self.streams.values():
            stream.close()
