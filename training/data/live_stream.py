"""Mix pinned upstream documents as training needs them, without a full-corpus gate.

Background workers fetch raw rows ahead of use, but only consumed rows advance the
checkpoint cursors. An append-only seen ledger prevents repeats and raises on newly
discovered split conflicts. Its configurable size cap bounds memory; large-scale
indexed ledgers are future work.
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
from .simple_llm import SPLIT_VERSION, adapt, validate_recipe
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

FETCH_WORKERS = 3
FETCH_ATTEMPTS = 4
CHUNK_ROWS = 16
CHUNK_TEXT_CHARACTERS = 4 * 1024**2
RETRYABLE_MODULES = ("requests", "urllib3", "httpx", "httpcore", "huggingface_hub")
END_OF_SHARD = None


class SplitConflict(ValueError):
    """A newly read document duplicates one already assigned to another split."""


def _retryable(exc):
    return isinstance(exc, (OSError, EOFError)) or type(exc).__module__.startswith(
        RETRYABLE_MODULES
    )


def _text_length(row):
    text = row.get("content", row.get("text", ""))
    return len(text) if isinstance(text, str) else 0


def _read_chunk(rows):
    """Read a bounded chunk of (index, row) pairs; also report whether the shard ended."""
    chunk, characters = [], 0
    while len(chunk) < CHUNK_ROWS and characters < CHUNK_TEXT_CHARACTERS:
        item = next(rows, None)
        if item is None:
            return chunk, True
        chunk.append(item)
        characters += _text_length(item[1])
    return chunk, False


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
            raise ValueError("Live data requires allow_unreviewed_data")
        if (
            self.manifest.get("tokenizer_sha256") != TOKENIZER_SHA256
            or self.manifest.get("max_passes") != 2
        ):
            raise ValueError("Tokenizer or repetition policy changed")
        if self.manifest.get("split_policy") != SPLIT_VERSION:
            raise ValueError("Split policy changed")
        if type(context) is not int or context < 1:
            raise ValueError("Context must be a positive integer")
        self.maximum_seen = self.manifest.get("max_seen_documents", 5_000_000)
        if type(self.maximum_seen) is not int or self.maximum_seen < 1:
            raise ValueError("Seen-document cap must be a positive integer")

        self.context, self.split = context, split
        self.max_passes = 2 if split == "train" else 1
        self.recipe = validate_recipe(copy.deepcopy(self.manifest["recipe"]))
        self.sources = {s["name"]: s for s in self.recipe["sources"]}
        self.identity = {
            "manifest_sha256": digest(self.path),
            "context": context,
            "split": split,
            "tokenizer_sha256": TOKENIZER_SHA256,
        }
        self.tokenizer = load_frozen()
        self.reader = reader or CachedShards()
        self.streams = {n: PersistentStream(self.reader) for n in self.sources}

        self.positions = dict.fromkeys(self.sources, 0)
        self.exposures = dict.fromkeys(self.sources, 0)
        self.cursors = {
            n: {"pass": 0, "shard": 0, "row": 0, "tokens": None, "offset": 0} for n in self.sources
        }
        self.events = []  # (content hash, split, first location), in discovery order.
        self.seen = {}
        self.iterators = {}
        self._start_fetching()

    # Raw rows: one in-flight chunk per source, filled ahead of the cursor.

    def _start_fetching(self):
        self.pool = ThreadPoolExecutor(max_workers=FETCH_WORKERS)
        self.pending = {}
        self.raw = {n: deque() for n in self.sources}
        self.started = False

    def _stop_fetching(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for stream in self.streams.values():
            stream.close()

    def _fetch(self, name, shard, start):
        source = self.sources[name]
        for attempt in range(FETCH_ATTEMPTS):
            rows = self.streams[name](source, source["files"][shard], start)
            try:
                return _read_chunk(rows)
            except Exception as exc:
                if not _retryable(exc) or attempt == FETCH_ATTEMPTS - 1:
                    raise
                self.streams[name].close()
                time.sleep(2**attempt)
            finally:
                rows.close()

    def _prefetch(self, name):
        cursor = self.cursors[name]
        if (
            name not in self.pending
            and not self.raw[name]
            and cursor["shard"] < len(self.sources[name]["files"])
        ):
            self.pending[name] = self.pool.submit(self._fetch, name, cursor["shard"], cursor["row"])

    def _next_row(self, name):
        """Return the next raw row and its ledger location, or None at the end of a pass."""
        if not self.started:
            for n in self.sources:
                self._prefetch(n)
            self.started = True
        cursor = self.cursors[name]
        while cursor["shard"] < len(self.sources[name]["files"]):
            if not self.raw[name]:
                self._prefetch(name)
                rows, ended = self.pending.pop(name).result()
                self.raw[name].extend(rows)
                if ended:
                    self.raw[name].append(END_OF_SHARD)
            item = self.raw[name].popleft()
            if item is END_OF_SHARD:
                cursor["shard"], cursor["row"] = cursor["shard"] + 1, 0
                continue
            index, row = item
            cursor["row"] = index + 1
            self._prefetch(name)
            return row, (name, cursor["shard"], index)
        return None

    # Documents: filter, deduplicate against the ledger, tokenize.

    def _admit(self, document, location, pass_index):
        """Record a document in the seen ledger; return whether this pass may use it."""
        key = hashlib.sha256(document["text"].encode()).digest()
        previous = self.seen.get(key)
        if previous is None:
            if len(self.events) >= self.maximum_seen:
                raise ValueError("Live seen-document ledger reached its configured memory cap")
            self.events.append((key, document["split"], location))
            self.seen[key] = (document["split"], location)
            return True
        split, owner = previous
        if split != document["split"]:
            raise SplitConflict(
                "New cross-split duplicate encountered; pause for corpus reconciliation"
            )
        # A repeat pass reuses each document only at the location it was first seen.
        return pass_index > 0 and owner == location

    def _next_document(self, name):
        """Return the next document's tokens with BOS/EOS, or None once passes are spent."""
        cursor = self.cursors[name]
        while True:
            item = self._next_row(name)
            if item is None:
                if cursor["pass"] + 1 >= self.max_passes:
                    return None
                cursor.update({"pass": cursor["pass"] + 1, "shard": 0, "row": 0})
                self.streams[name].close()
                continue
            row, location = item
            source = self.sources[name]
            document, _reason = adapt(source, row, (source["files"][location[1]], location[2]))
            if document is None or not self._admit(document, location, cursor["pass"]):
                continue
            if document["split"] != self.split:
                continue
            ids = self.tokenizer.encode(document["text"], add_special_tokens=False).ids
            if ids and min(ids) >= FIRST_CONTENT_TOKEN_ID:
                return (BOS_TOKEN_ID, *ids, EOS_TOKEN_ID)

    # Windows and batches: document-local, padded, chosen by exposure deficit.

    def blocks(self, name):
        cursor = self.cursors[name]
        while True:
            if cursor["tokens"] is None:
                cursor["tokens"], cursor["offset"] = self._next_document(name), 0
                if cursor["tokens"] is None:
                    return
            tokens, offset = cursor["tokens"], cursor["offset"]
            labels = list(tokens[offset + 1 : offset + self.context + 1])
            count = len(labels)
            inputs = list(tokens[offset : offset + count])
            padding = self.context - count
            cursor["offset"] += self.context
            if cursor["offset"] >= len(tokens) - 1:
                cursor["tokens"], cursor["offset"] = None, 0
            yield {
                "input_ids": inputs + [PAD_TOKEN_ID] * padding,
                "labels": labels + [LOSS_IGNORE_INDEX] * padding,
                "loss_tokens": count,
                "source": name,
                "pass": cursor["pass"] + 1,
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
        except Exception:
            self.load_state_dict(self.state_at(before))
            raise
        return (
            torch.tensor([b["input_ids"] for b in blocks], dtype=torch.long, device=device),
            torch.tensor([b["labels"] for b in blocks], dtype=torch.long, device=device),
        )

    # Checkpoint state: cursors plus the ledger prefix they had seen.

    def cursor_state(self):
        return {
            **self.identity,
            "positions": dict(self.positions),
            "loss_token_exposures": dict(self.exposures),
            "live_cursors": {n: dict(c) for n, c in self.cursors.items()},
            "seen_count": len(self.events),
        }

    def state_at(self, cursor):
        return {**copy.deepcopy(cursor), "seen_events": self.events[: cursor["seen_count"]]}

    def state_dict(self):
        return self.state_at(self.cursor_state())

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Live checkpoint identity changed")
        cursors = state["live_cursors"]
        if set(cursors) != set(self.sources) or state["seen_count"] != len(state["seen_events"]):
            raise ValueError("Invalid live checkpoint cursors")
        for c in cursors.values():
            if c["pass"] not in (0, 1) or min(c["shard"], c["row"], c["offset"]) < 0:
                raise ValueError("Invalid live cursor")
        self._stop_fetching()
        self.cursors = copy.deepcopy(cursors)
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.events = list(state["seen_events"])
        self.seen = {key: (split, tuple(owner)) for key, split, owner in self.events}
        self.iterators = {}
        self._start_fetching()

    def close(self):
        self._stop_fetching()
