import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from training_data.heldout import load_heldout
from training_data.prepared_reference import DOCUMENTS, build
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import load_frozen


def database(folder, source, documents):
    """documents: (split, parent, text) rows stored under `source`."""
    folder.mkdir()
    path = folder / "raw.sqlite"
    db = sqlite3.connect(path)
    db.execute(DOCUMENTS)
    tokenizer = load_frozen()
    for split, parent, text in documents:
        db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
            (
                hashlib.sha256(text.encode()).hexdigest(),
                source,
                split,
                parent,
                json.dumps({"language": "python"}),
                zlib.compress(text.encode()),
                len(tokenizer.encode(text, add_special_tokens=False).ids) + 1,
            ),
        )
    db.commit()
    db.close()
    return path


class PreparedReferenceTests(unittest.TestCase):
    def test_references_exclude_heldout_parents_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = database(
                root / "code",
                "code_python",
                [
                    ("train", "repo/a", "def train_a(): return 1 " * 20),
                    ("validation", "repo/b", "def held_b(): return 2 " * 20),
                ],
            )
            # A training file from the held-out repository in another database is hidden.
            patches = database(
                root / "patches",
                "code",
                [
                    ("train", "repo/b", "def leaked_b(): return 3 " * 20),
                    ("train", "repo/c", "def train_c(): return 4 " * 20),
                    ("validation", "repo/d", "def held_d(): return 5 " * 20),
                ],
            )
            books = database(
                root / "books",
                "books",
                [
                    ("train", "book/1", "Once upon a time there was a parser. " * 20),
                    ("validation", "book/2", "A held-out chapter about tokens. " * 20),
                ],
            )
            databases = [
                (code, {"code_python": "code"}),
                (patches, {"code": "code"}),
                (books, {"books": "books"}),
            ]
            manifest = build(
                root / "snapshot", {"code": 60, "books": 40}, databases, target=50, context=8
            )
            self.assertEqual(manifest["excluded_parents"], ["book/2", "repo/b", "repo/d"])
            self.assertEqual(manifest["heldout"]["windows"], 3)
            path = root / "snapshot/manifest.json"
            data = RawTokenMixture(path, context=8, allow_unreviewed=True)
            hidden = data.db.execute(
                "SELECT COUNT(*) FROM documents WHERE parent='repo/b' AND split='train'"
            ).fetchone()[0]
            self.assertEqual(hidden, 0)
            counted = data.db.execute(
                "SELECT SUM(loss_tokens) FROM documents WHERE source='code' AND split='train'"
            ).fetchone()[0]
            self.assertEqual(manifest["sources"]["code"]["splits"]["train"]["loss_tokens"], counted)
            heldout = load_heldout(path, context=8)
            self.assertEqual({b["stratum"] for b in heldout["code"]}, {"code/python/sample"})
            data.next_block()
            state = data.state_dict()
            expected = data.next_block()
            restored = RawTokenMixture(path, context=8, allow_unreviewed=True)
            restored.load_state_dict(state)
            self.assertEqual(restored.next_block(), expected)
            data.close()
            restored.close()
            self.assertLess((root / "snapshot/raw.sqlite").stat().st_size, 100_000)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                build(root / "snapshot", {"code": 60, "books": 40}, databases, target=50, context=8)
            with self.assertRaisesRegex(ValueError, "two-pass capacity"):
                build(root / "large", {"code": 60, "books": 40}, databases, target=10**9)
            with self.assertRaisesRegex(ValueError, "share documents"):
                build(root / "twice", {"code": 100}, [databases[0], databases[0]], target=10)


if __name__ == "__main__":
    unittest.main()
