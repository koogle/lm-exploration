"""Real-file cursor and format checks; no GPU startup emulation."""

import copy
import gzip
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data.subword import TOKENIZER_SHA256, digest
from training_data.token_shards import (
    FORMAT,
    HEADER,
    MAGIC,
    convert_snapshot,
    file_entry,
    verify_snapshot,
    write_shard,
)
from training_data.token_stream import TokenMixture


class SeekableTokenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            {
                "ids": list(range(4, 4 + length)),
                "hash": hashlib.sha256(str(i).encode()).hexdigest(),
                "parent": f"parent-{i}",
                "provenance": {"source": "fixture"},
            }
            for i, length in enumerate((1, 8, 9, 38))
        ]
        (self.root / "audit.json").write_text("{}")
        shards = [
            write_shard(self.root / f"{i}.tokens", self.rows[i * 2 : i * 2 + 2]) for i in range(2)
        ]
        self.manifest = {
            "schema": 2,
            "token_format": FORMAT,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "training_ready": False,
            "max_passes": 2,
            "audit": file_entry(self.root / "audit.json"),
            "sources": {
                "fixture": {
                    "weight_percent": 100,
                    "splits": {
                        split: {
                            "loss_tokens": sum(len(r["ids"]) + 1 for r in self.rows),
                            "shards": shards,
                        }
                        for split in ("train", "validation")
                    },
                }
            },
        }
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))

    def loader(self, **kwargs):
        loader = TokenMixture(self.path, allow_unreviewed=True, **kwargs)
        self.addCleanup(lambda: [it.close() for it in loader.iterators.values()])
        return loader

    def expected(self, context, passes):
        for pass_index in range(passes):
            for row in self.rows:
                tokens = [1, *row["ids"], 2]
                for offset in range(0, len(tokens) - 1, context):
                    count = min(context, len(tokens) - 1 - offset)
                    yield {
                        "input_ids": tokens[offset : offset + count] + [0] * (context - count),
                        "labels": tokens[offset + 1 : offset + count + 1]
                        + [-100] * (context - count),
                        "loss_tokens": count,
                        "source": "fixture",
                        "pass": pass_index + 1,
                        "document_hash": row["hash"],
                    }

    def test_windows_and_every_resume_boundary_match_reference(self):
        for context in (1, 4, 8, 64):
            for split, passes in (("train", 2), ("validation", 1)):
                with self.subTest(context=context, split=split):
                    loader = self.loader(context=context, split=split)
                    for expected in self.expected(context, passes):
                        saved = json.loads(json.dumps(loader.state_dict()))
                        restored = self.loader(context=context, split=split)
                        restored.load_state_dict(saved)
                        self.assertEqual(restored.next_block(), expected)
                        self.assertEqual(loader.next_block(), expected)
                        self.assertEqual(restored.state_dict(), loader.state_dict())
                        for it in restored.iterators.values():
                            it.close()
                    saved = loader.state_dict()
                    restored.load_state_dict(saved)
                    with self.assertRaises(StopIteration):
                        restored.next_block()
                    self.assertEqual(restored.state_dict(), saved)

    def test_startup_only_hashes_manifest_and_explicit_verification_detects_corruption(self):
        with patch("training_data.token_stream.digest", wraps=digest) as hashing:
            self.loader()
        self.assertEqual([call.args[0] for call in hashing.call_args_list], [self.path])
        verify_snapshot(self.path)
        # A valid-token, same-size edit cannot be detected by cheap startup metadata checks.
        with (self.root / "0.tokens").open("r+b") as handle:
            handle.seek(len(MAGIC) + HEADER.size)
            handle.write(struct.pack("<H", 7))
        self.loader()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_snapshot(self.path)

    def test_resume_does_not_read_consumed_shards_or_document_prefix(self):
        loader = self.loader(context=4)
        while True:
            loader.next_block()
            state = loader.state_dict()
            cursor = state["cursors"]["fixture"]
            if cursor["shard"] == 1 and cursor["token"] == 4:
                break
        expected = loader.next_block()
        for it in loader.iterators.values():
            it.close()
        # Make consumed data unreadable without changing file sizes. A replay would fail.
        previous = self.root / "0.tokens"
        previous.write_bytes(b"!" * previous.stat().st_size)
        with (self.root / "1.tokens").open("r+b") as handle:
            handle.seek(cursor["byte"] + HEADER.size)
            handle.write(b"\xff\xff")
        restored = self.loader(context=4)
        restored.load_state_dict(state)
        self.assertEqual(restored.next_block(), expected)

    def test_cursor_rejection_is_atomic_and_size_changes_fail_at_startup(self):
        loader = self.loader(context=4)
        loader.next_block()
        good = loader.state_dict()
        for key, value in (("pass", 3), ("shard", 9), ("byte", 0), ("token", 3)):
            bad = copy.deepcopy(good)
            bad["cursors"]["fixture"][key] = value
            with self.assertRaisesRegex(ValueError, "Invalid seekable data cursor"):
                loader.load_state_dict(bad)
            self.assertEqual(loader.state_dict(), good)
        with (self.root / "0.tokens").open("ab") as handle:
            handle.write(b"extra")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.loader()

    def test_conversion_preserves_windows_provenance_and_original(self):
        legacy = copy.deepcopy(self.manifest)
        legacy.pop("token_format")
        legacy["schema"] = 1
        path = self.root / "legacy.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        for split in legacy["sources"]["fixture"]["splits"].values():
            split["shards"] = [file_entry(path)]
        # Use separate files for each split, as a real export does.
        validation = self.root / "validation.jsonl.gz"
        validation.write_bytes(path.read_bytes())
        legacy["sources"]["fixture"]["splits"]["validation"]["shards"] = [file_entry(validation)]
        self.path.write_text(json.dumps(legacy))
        before = digest(self.path), digest(path)
        with self.assertRaisesRegex(ValueError, "convert legacy snapshots"):
            self.loader()
        destination = self.root / "converted"
        converted = convert_snapshot(self.path, destination)
        self.assertEqual((digest(self.path), digest(path)), before)
        self.assertFalse(converted["training_ready"])
        verify_snapshot(destination / "manifest.json")
        restored = TokenMixture(destination / "manifest.json", context=4, allow_unreviewed=True)
        self.assertEqual(list(restored.blocks("fixture")), list(self.expected(4, 2)))
        with gzip.open(destination / "legacy.provenance.jsonl.gz", "rt") as handle:
            metadata = [json.loads(line) for line in handle]
        self.assertEqual(
            metadata,
            [
                {**{k: v for k, v in row.items() if k != "ids"}, "content_tokens": len(row["ids"])}
                for row in self.rows
            ],
        )
        with self.assertRaises(FileExistsError):
            convert_snapshot(self.path, destination)


if __name__ == "__main__":
    unittest.main()
