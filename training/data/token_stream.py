"""Deterministic token-weighted batches from a frozen preparation snapshot."""

import gzip
import itertools
import json
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


class TokenMixture:
    """One source per window, chosen by actual loss-token exposure deficit.

    Ordering is deterministic (content-hash order), with no implicit RNG or shuffle.
    Save state_dict in the model checkpoint at the optimizer boundary. Restoring
    replays compressed shards to the saved block offsets. Exhaustion stops training;
    it never silently changes source weights or exceeds two passes.
    """

    def __init__(self, manifest_path, context=2048, split="train", allow_unreviewed=False):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        if context < 1 or split not in ("train", "validation"):
            raise ValueError("Invalid context or split")
        if self.manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Tokenizer identity mismatch")
        from .review import require_data_review

        require_data_review(self.manifest["training_ready"], allow_unreviewed)
        if self.manifest["max_passes"] != 2:
            raise ValueError("Unexpected repetition policy")
        self.context, self.split = context, split
        self.identity = {
            "manifest_sha256": digest(self.path),
            "context": context,
            "split": split,
            "tokenizer_sha256": TOKENIZER_SHA256,
        }
        self.sources = self.manifest["sources"]
        self.positions = dict.fromkeys(self.sources, 0)
        self.exposures = dict.fromkeys(self.sources, 0)
        self.iterators = {}
        audit = self.manifest["audit"]
        if (
            Path(audit["path"]).name != audit["path"]
            or digest(self.path.parent / audit["path"]) != audit["sha256"]
        ):
            raise ValueError("Audit path or checksum mismatch")
        for name, source in self.sources.items():
            if source["splits"][split]["loss_tokens"] <= 0:
                raise ValueError(f"Source {name} has no {split} tokens; cannot honor mixture")
            for shard in source["splits"][split]["shards"]:
                path = self.path.parent / shard["path"]
                if Path(shard["path"]).name != shard["path"] or digest(path) != shard["sha256"]:
                    raise ValueError("Shard path or checksum mismatch")

    def blocks(self, name):
        source = self.sources[name]["splits"][self.split]
        passes = 2 if self.split == "train" else 1
        for pass_index in range(passes):
            for shard in source["shards"]:
                with gzip.open(self.path.parent / shard["path"], "rt") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if not row["ids"] or any(
                            type(t) is not int or t < FIRST_CONTENT_TOKEN_ID or t >= VOCAB_SIZE
                            for t in row["ids"]
                        ):
                            raise ValueError("Invalid content token IDs")
                        tokens = [BOS_TOKEN_ID, *row["ids"], EOS_TOKEN_ID]
                        for offset in range(0, len(tokens) - 1, self.context):
                            x = tokens[offset : offset + self.context]
                            y = tokens[offset + 1 : offset + self.context + 1]
                            count = len(y)
                            # At the end, x can include EOS without a next target.
                            x = x[:count]
                            yield {
                                "input_ids": x + [PAD_TOKEN_ID] * (self.context - count),
                                "labels": y + [LOSS_IGNORE_INDEX] * (self.context - count),
                                "loss_tokens": count,
                                "source": name,
                                "pass": pass_index + 1,
                                "document_hash": row["hash"],
                            }

    def next_block(self):
        name = min(
            self.sources, key=lambda n: (self.exposures[n] / self.sources[n]["weight_percent"], n)
        )
        if name not in self.iterators:
            self.iterators[name] = itertools.islice(self.blocks(name), self.positions[name], None)
        try:
            block = next(self.iterators[name])
        except StopIteration as exc:
            raise StopIteration(f"{name} exhausted its allowed passes; expand capacity") from exc
        self.positions[name] += 1
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
        self.positions = dict(state["positions"])
        self.exposures = dict(state["loss_token_exposures"])
        self.iterators.clear()
