import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch
from training_data.mixture import check_capacity
from training_data.raw_token_stream import RawTokenMixture, token_targets
from training_data.subword import LOSS_IGNORE_INDEX, TOKENIZER_SHA256, digest, load_frozen

SCHEMA = """CREATE TABLE documents (
    hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
    provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0)"""


def snapshot(root, sources):
    """One train and one validation document per source; returns train target counts."""
    db = sqlite3.connect(root / "raw.sqlite")
    db.execute(SCHEMA)
    counts = {}
    for name in sources:
        for split in ("train", "validation"):
            text = f"{name} {split} keeps each token and document boundary. " * 20
            count = len(load_frozen().encode(text, add_special_tokens=False).ids) + 1
            key = hashlib.sha256(text.encode()).hexdigest()
            db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
                (key, name, split, name, "{}", zlib.compress(text.encode()), count),
            )
            counts[name, split] = count
    db.commit()
    db.close()
    manifest = {
        "format": "raw-text-v1",
        "training_ready": True,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")},
        "sources": {
            name: {
                "weight_percent": 100 / len(sources),
                **extra,
                "splits": {
                    split: {"loss_tokens": counts[name, split]} for split in ("train", "validation")
                },
            }
            for name, extra in sources.items()
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, {name: counts[name, "train"] for name in sources}


def drain(mixture):
    total = 0
    while True:
        try:
            total += mixture.next_block()["loss_tokens"]
        except StopIteration:
            return total


class SourceBudgetTests(unittest.TestCase):
    def test_token_targets_are_exact_and_deterministic(self):
        targets = token_targets({"code": 60, "web": "22.5", "books": 17.5}, 1_000_001)
        self.assertEqual(sum(targets.values()), 1_000_001)
        self.assertEqual(targets, {"code": 600_001, "web": 225_000, "books": 175_000})
        with self.assertRaises(ValueError):
            token_targets({"code": 60}, 10)

    def test_one_pass_cap_survives_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path, counts = snapshot(Path(folder), {"code": {"max_passes": 1}, "web": {}})
            a = RawTokenMixture(path, context=8)
            self.assertEqual(a.passes, {"code": 1, "web": 2})
            a.next_block()
            b = RawTokenMixture(path, context=8)
            b.load_state_dict(a.state_dict())
            first = a.state_dict()["loss_token_exposures"]
            rest = sum(block["loss_tokens"] for block in b.blocks("code"))
            self.assertEqual(sum(first.values()) + rest, counts["code"])
            bad = a.state_dict()
            bad["raw_cursors"]["code"]["pass"] = 1
            with self.assertRaisesRegex(ValueError, "Invalid raw cursor"):
                a.load_state_dict(bad)
            a.close()
            b.close()

    def test_exact_targets_mask_excess_pad_final_batch_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = snapshot(
                Path(folder),
                {"code": {"max_passes": 1, "target_tokens": 17}, "web": {"target_tokens": 11}},
            )
            a = RawTokenMixture(path, context=8)
            _, y = a.next_batch(4)
            first = int((y != LOSS_IGNORE_INDEX).sum())
            state = a.state_dict()
            expected = a.next_batch(4)
            b = RawTokenMixture(path, context=8)
            b.load_state_dict(state)
            actual = b.next_batch(4)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(first + int((actual[1] != LOSS_IGNORE_INDEX).sum()), 28)
            self.assertEqual(b.exposures, {"code": 17, "web": 11})
            self.assertTrue((actual[1][-1] == LOSS_IGNORE_INDEX).all())
            with self.assertRaises(StopIteration):
                b.next_batch(4)
            self.assertEqual(a.state_dict(), b.state_dict())
            over = b.state_dict()
            over["loss_token_exposures"]["code"] = 18
            with self.assertRaisesRegex(ValueError, "exceeds a source target"):
                b.load_state_dict(over)
            a.close()
            b.close()

    def test_targets_must_fit_pass_caps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, counts = snapshot(root, {"code": {"max_passes": 1}})
            manifest = json.loads(path.read_text())
            manifest["sources"]["code"]["target_tokens"] = counts["code"] + 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "within its pass capacity"):
                RawTokenMixture(path, context=8)

    def test_capacity_check_uses_source_caps_and_targets(self):
        def mixture(**sources):
            return SimpleNamespace(manifest={"max_passes": 2}, sources=sources)

        def source(weight, train, **extra):
            return {"weight_percent": weight, "splits": {"train": {"loss_tokens": train}}, **extra}

        check_capacity(mixture(a=source(50, 100), b=source(50, 100)), 400)
        with self.assertRaisesRegex(ValueError, "pass caps"):
            check_capacity(mixture(a=source(50, 100, max_passes=1), b=source(50, 100)), 400)
        exact = mixture(a=source(50, 100, target_tokens=150), b=source(50, 100, target_tokens=50))
        check_capacity(exact, 200)
        with self.assertRaisesRegex(ValueError, "sum to the token target"):
            check_capacity(exact, 201)
        with self.assertRaisesRegex(ValueError, "within its pass cap"):
            capped = source(50, 100, target_tokens=150, max_passes=1)
            check_capacity(mixture(a=capped, b=source(50, 100, target_tokens=50)), 200)


if __name__ == "__main__":
    unittest.main()
