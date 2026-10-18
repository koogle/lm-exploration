"""Deterministic token-weighted batches from a frozen preparation snapshot."""

import copy
import json
import struct
from pathlib import Path

from .subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    FIRST_CONTENT_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    PAD_TOKEN_ID,
    TOKENIZER_SHA256,
    VOCAB_SIZE,
    digest,
)
from .token_shards import FORMAT, HEADER, MAGIC, snapshot_file


class TokenMixture:
    """One source per window, chosen by actual loss-token exposure deficit.

    Ordering is deterministic (content-hash order), with no implicit RNG or shuffle.
    Save state_dict in the model checkpoint at the optimizer boundary. Restoring
    seeks directly to the saved document and token offsets. Exhaustion stops training;
    it never silently changes source weights or exceeds two passes.
    """

    def __init__(self, manifest_path, context=2048, split="train", allow_unreviewed=False):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        if context < 1 or split not in ("train", "validation"):
            raise ValueError("Invalid context or split")
        if self.manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Tokenizer identity mismatch")
        if self.manifest.get("token_format") != FORMAT or self.manifest.get("schema") != 2:
            raise ValueError(
                "Seekable token snapshot required; convert legacy snapshots with "
                "python -m training_data.token_shards convert MANIFEST NEW_DIRECTORY"
            )
        if not self.manifest["training_ready"] and not allow_unreviewed:
            raise ValueError("Preparation snapshot has pending audits; pilot override required")
        if self.manifest["max_passes"] != 2:
            raise ValueError("Unexpected repetition policy")
        self.context, self.split = context, split
        self.identity = {
            "manifest_sha256": digest(self.path),
            "context": context,
            "split": split,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "cursor_format": FORMAT,
        }
        self.sources = self.manifest["sources"]
        self.positions = dict.fromkeys(self.sources, 0)
        self.cursors = {name: self._cursor() for name in self.sources}
        self.exposures = dict.fromkeys(self.sources, 0)
        self.iterators = {}
        # Snapshots are immutable. Full checksums are checked explicitly at preparation
        # or transfer time; startup reads the manifest and stats files, not their bytes.
        snapshot_file(self.path.parent, self.manifest["audit"])
        for name, source in self.sources.items():
            if source["splits"][split]["loss_tokens"] <= 0:
                raise ValueError(f"Source {name} has no {split} tokens; cannot honor mixture")
            for shard in source["splits"][split]["shards"]:
                if type(shard.get("size_bytes")) is not int or shard["size_bytes"] <= len(MAGIC):
                    raise ValueError("Invalid token shard size")
                snapshot_file(self.path.parent, shard)

    @staticmethod
    def _cursor(pass_index=0, shard=0, byte=None, token=0):
        return {
            "pass": pass_index,
            "shard": shard,
            "byte": len(MAGIC) if byte is None else byte,
            "token": token,
        }

    def _advance(self, shards, pass_index, shard, byte, token):
        if byte == shards[shard]["size_bytes"]:
            shard, byte, token = shard + 1, len(MAGIC), 0
            if shard == len(shards):
                pass_index, shard = pass_index + 1, 0
        return self._cursor(pass_index, shard, byte, token)

    def blocks(self, name):
        """Iterate a source from the beginning without changing its training cursor."""
        for block, _ in self._blocks(name, self._cursor()):
            yield block

    def _blocks(self, name, cursor):
        shards = self.sources[name]["splits"][self.split]["shards"]
        passes = 2 if self.split == "train" else 1
        while cursor["pass"] < passes:
            pass_index, shard_index = cursor["pass"], cursor["shard"]
            shard = shards[shard_index]
            with snapshot_file(self.path.parent, shard).open("rb") as handle:
                if handle.read(len(MAGIC)) != MAGIC:
                    raise ValueError("Invalid token shard header")
                while cursor["pass"] == pass_index and cursor["shard"] == shard_index:
                    byte, offset = cursor["byte"], cursor["token"]
                    handle.seek(byte)
                    header = handle.read(HEADER.size)
                    if len(header) != HEADER.size:
                        raise ValueError("Truncated document header")
                    length, content_hash = HEADER.unpack(header)
                    end = byte + HEADER.size + 2 * length
                    if not length or end > shard["size_bytes"] or offset > length:
                        raise ValueError("Invalid document length or token cursor")
                    while offset <= length:
                        count = min(self.context, length + 1 - offset)
                        # Read only this window's content, adding virtual BOS/EOS.
                        first, last = max(1, offset), min(length, offset + count)
                        handle.seek(byte + HEADER.size + 2 * (first - 1))
                        raw = handle.read(2 * (last - first + 1))
                        if len(raw) != 2 * (last - first + 1):
                            raise ValueError("Truncated token window")
                        tokens = list(struct.unpack(f"<{last - first + 1}H", raw))
                        if any(t < FIRST_CONTENT_TOKEN_ID or t >= VOCAB_SIZE for t in tokens):
                            raise ValueError("Invalid content token IDs")
                        if offset == 0:
                            tokens.insert(0, BOS_TOKEN_ID)
                        if offset + count == length + 1:
                            tokens.append(EOS_TOKEN_ID)
                        offset += count
                        cursor = self._advance(
                            shards,
                            pass_index,
                            shard_index,
                            end if offset == length + 1 else byte,
                            0 if offset == length + 1 else offset,
                        )
                        yield (
                            {
                                "input_ids": tokens[:-1] + [PAD_TOKEN_ID] * (self.context - count),
                                "labels": tokens[1:] + [LOSS_IGNORE_INDEX] * (self.context - count),
                                "loss_tokens": count,
                                "source": name,
                                "pass": pass_index + 1,
                                "document_hash": content_hash.hex(),
                            },
                            cursor,
                        )

    def next_block(self):
        name = min(
            self.sources, key=lambda n: (self.exposures[n] / self.sources[n]["weight_percent"], n)
        )
        if name not in self.iterators:
            self.iterators[name] = self._blocks(name, self.cursors[name])
        try:
            block, cursor = next(self.iterators[name])
        except StopIteration as exc:
            raise StopIteration(f"{name} exhausted its allowed passes; expand capacity") from exc
        self.positions[name] += 1
        self.cursors[name] = cursor
        self.exposures[name] += block["loss_tokens"]
        return block

    def next_batch(self, batch_size, device="cpu"):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # A failed partial batch must not consume data without a training update.
        state = self.state_dict()
        try:
            blocks = [self.next_block() for _ in range(batch_size)]
            x = torch.tensor([b["input_ids"] for b in blocks], dtype=torch.long, device=device)
            y = torch.tensor([b["labels"] for b in blocks], dtype=torch.long, device=device)
        except Exception:
            self.load_state_dict(state)
            raise
        return x, y

    def state_dict(self):
        return {
            **self.identity,
            "positions": dict(self.positions),
            "cursors": copy.deepcopy(self.cursors),
            "loss_token_exposures": dict(self.exposures),
        }

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Checkpoint data identity mismatch")
        for key in ("positions", "loss_token_exposures"):
            if set(state[key]) != set(self.sources) or any(
                type(v) is not int or v < 0 for v in state[key].values()
            ):
                raise ValueError("Invalid data cursor")
        cursors = state.get("cursors", {})
        if set(cursors) != set(self.sources):
            raise ValueError("Missing seekable data cursors")
        passes = 2 if self.split == "train" else 1
        for name, cursor in cursors.items():
            shards = self.sources[name]["splits"][self.split]["shards"]
            if (
                not isinstance(cursor, dict)
                or set(cursor) != {"pass", "shard", "byte", "token"}
                or any(type(v) is not int or v < 0 for v in cursor.values())
                or cursor["pass"] > passes
                or cursor["shard"] >= len(shards)
                or cursor["token"] % self.context
                or not len(MAGIC) <= cursor["byte"] < shards[cursor["shard"]]["size_bytes"]
                or (cursor["pass"] == passes and cursor != self._cursor(passes))
            ):
                raise ValueError("Invalid seekable data cursor")
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.cursors = copy.deepcopy(cursors)
        for iterator in self.iterators.values():
            iterator.close()
        self.iterators.clear()
