import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

from distillation.data import CodeWindows


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) for c in text])


class DataTests(unittest.TestCase):
    def test_resume_splits_exclusions_and_one_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = root / "raw.sqlite"
            manifest = root / "manifest.json"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE documents (hash TEXT PRIMARY KEY, parent TEXT, text BLOB, loss_tokens INTEGER, source TEXT, split TEXT, conflict INTEGER)"
            )
            for key, parent, text, split, conflict in [
                ("a", "excluded", "bad", "train", 0),
                ("b", "good", "abcdef", "train", 0),
                ("c", "good", "bad", "train", 1),
                ("d", "good", "val", "validation", 0),
                ("e", "good", "excluded-hash", "train", 0),
            ]:
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                    (
                        key,
                        parent,
                        zlib.compress(text.encode()),
                        len(text) + 1,
                        "code",
                        split,
                        conflict,
                    ),
                )
            conn.commit()
            conn.close()
            manifest.write_text(
                json.dumps(
                    {
                        "external_databases": [
                            {
                                "path": "/remote/raw.sqlite",
                                "size_bytes": db.stat().st_size,
                                "sources": {"code": "code"},
                            }
                        ],
                        "excluded_validation_hashes": ["e"],
                        "excluded_validation_parents": ["excluded"],
                    }
                )
            )

            def loader(**kwargs):
                return CodeWindows(
                    manifest, db, Tokenizer(), context=3, root=root, **kwargs
                )

            first = loader()
            self.assertEqual(first.next(), ([1, 97, 98], [97, 98, 99]))
            resumed = loader(state=first.state)
            self.assertEqual(first.next(), resumed.next())
            self.assertEqual(first.next(), resumed.next())
            with self.assertRaises(StopIteration):
                first.next()
            with self.assertRaises(StopIteration):
                resumed.next()
            val = loader(split="validation")
            self.assertEqual(val.next()[0], [1, 118, 97])
            # All training targets, including EOS, occur exactly once.
            all_data = loader()
            targets = []
            try:
                while True:
                    targets.extend(all_data.next()[1])
            except StopIteration:
                pass
            self.assertEqual(targets, [97, 98, 99, 100, 101, 102, 2])
