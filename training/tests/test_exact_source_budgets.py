import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

import torch
from training_data.overnight_corpus import SCHEMA
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import LOSS_IGNORE_INDEX, TOKENIZER_SHA256, digest, load_frozen


class ExactBudgetTests(unittest.TestCase):
    def test_collected_files_precede_supplements_and_restore_across_tiers(self):
        from training_data.python_experiment import Corpus
        from training_data.subword import BOS_TOKEN_ID

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = Corpus(root)
            collected = "recent complete repository source with its comments preserved " * 3
            key = hashlib.sha256(collected.encode()).hexdigest()
            older = next(
                f"older supplemental document {i} with useful content " * 3
                for i in range(1000)
                if hashlib.sha256(
                    (f"older supplemental document {i} with useful content " * 3).encode()
                ).hexdigest()
                < key
            )
            count = corpus.add(
                collected, "code", "org/recent", {"origin": "collected_git"}, split="train"
            )
            count += corpus.add(older, "code", "org/older", {"origin": "supplement"}, split="train")
            corpus.db.commit()
            corpus.db.close()
            manifest = {
                "format": "scholarly-raw-v1",
                "training_ready": True,
                "max_passes": 2,
                "prefer_collected": True,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "sources": {
                    "code": {
                        "weight_percent": 100,
                        "max_passes": 1,
                        "splits": {
                            "train": {"loss_tokens": count},
                            "validation": {"loss_tokens": count},
                        },
                    }
                },
                "database": {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            a = RawTokenMixture(path, context=8)
            first = a.next_block()
            expected_prefix = [
                BOS_TOKEN_ID,
                *load_frozen().encode(collected, add_special_tokens=False).ids,
            ][:8]
            self.assertEqual(first["input_ids"], expected_prefix)
            b = RawTokenMixture(path, context=8)
            b.load_state_dict(a.state_dict())
            while True:
                try:
                    expected = a.next_block()
                except StopIteration:
                    break
                self.assertEqual(b.next_block(), expected)
            with self.assertRaises(StopIteration):
                b.next_block()
            self.assertEqual(b.exposures["code"], count)
            a.close()
            b.close()

    def test_exact_category_targets_partial_batch_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = sqlite3.connect(root / "raw.sqlite")
            db.executescript(SCHEMA)
            sources = {}
            for name, budget in [("implementation", 17), ("documentation", 11)]:
                text = (name + " preserves each token and document boundary. ") * 20
                count = len(load_frozen().encode(text, add_special_tokens=False).ids) + 1
                key = hashlib.sha256(text.encode()).hexdigest()
                db.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
                    (key, name, "train", name, "{}", zlib.compress(text.encode()), count),
                )
                sources[name] = {
                    "weight_percent": 50,
                    "max_passes": 1,
                    "target_tokens": budget,
                    "splits": {
                        "train": {"loss_tokens": count},
                        "validation": {"loss_tokens": count},
                    },
                }
            db.commit()
            db.close()
            manifest = {
                "format": "scholarly-raw-v1",
                "training_ready": True,
                "max_passes": 2,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "sources": sources,
                "database": {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            a = RawTokenMixture(path, context=8)
            _, y = a.next_batch(4)
            first_count = int((y != LOSS_IGNORE_INDEX).sum())
            state = a.state_dict()
            expected = a.next_batch(4)
            b = RawTokenMixture(path, context=8)
            b.load_state_dict(state)
            actual = b.next_batch(4)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(first_count + int((actual[1] != LOSS_IGNORE_INDEX).sum()), 28)
            self.assertEqual(b.exposures, {"implementation": 17, "documentation": 11})
            self.assertTrue((actual[1][1:] == LOSS_IGNORE_INDEX).all())
            with self.assertRaises(StopIteration):
                b.next_batch(4)
            self.assertEqual(a.state_dict(), b.state_dict())
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
