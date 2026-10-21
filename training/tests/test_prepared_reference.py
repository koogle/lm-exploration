import hashlib
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from training_data.multilingual_code import SCHEMA
from training_data.prepared_reference import build
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import load_frozen

from trainer.representative_validation import load_heldout


class PreparedSnapshotTests(unittest.TestCase):
    def test_external_sources_validation_exclusion_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = load_frozen()
            paths, counts = {}, {}
            for kind, source in [
                ("code", "code"),
                ("patches", "code"),
                ("original", "books"),
                ("eu", "eu"),
            ]:
                folder = root / kind
                folder.mkdir()
                paths[kind] = folder / "raw.sqlite"
                db = sqlite3.connect(paths[kind])
                db.executescript(SCHEMA)
                count = 0
                for split in ["train", "validation"]:
                    text = f"{kind} {split} document words here " * 30
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                    count = tokens if split == "train" else count
                    db.execute(
                        "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
                        (
                            hashlib.sha256(text.encode()).hexdigest(),
                            source,
                            split,
                            kind + split,
                            "{}",
                            zlib.compress(text.encode()),
                            tokens,
                        ),
                    )
                # Cross-source training record from a sampled held-out parent is excluded.
                if kind == "patches":
                    text = "a different version of validation code"
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                    db.execute(
                        "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
                        (
                            "leaked",
                            source,
                            "train",
                            "codevalidation",
                            "{}",
                            zlib.compress(text.encode()),
                            tokens,
                        ),
                    )
                db.commit()
                db.close()
                counts[kind] = count
            manifest = build(
                root / "snapshot",
                {"code": 60, "books": 35, "eu": 5},
                [
                    (paths["code"], {"code": "code"}),
                    (paths["patches"], {"code": "code"}),
                    (paths["original"], {"books": "books"}),
                    (paths["eu"], {"eu": "eu"}),
                ],
                {
                    "code": counts["code"] + counts["patches"],
                    "books": counts["original"],
                    "eu": counts["eu"],
                },
                target=100,
                pass_caps={"code": 1},
            )
            self.assertEqual(manifest["sources"]["code"]["max_passes"], 1)
            data = RawTokenMixture(
                root / "snapshot/manifest.json", context=8, allow_unreviewed=True
            )
            self.assertEqual(
                data.db.execute("SELECT COUNT(*) FROM documents WHERE hash='leaked'").fetchone()[0],
                0,
            )
            self.assertEqual(
                {name for name, _ in load_heldout(data.path, data)}, {"code", "books", "eu"}
            )
            data.next_block()
            state = data.state_dict()
            expected = data.next_block()
            restored = RawTokenMixture(data.path, context=8, allow_unreviewed=True)
            restored.load_state_dict(state)
            self.assertEqual(restored.next_block(), expected)
            self.assertFalse(manifest["audit_policy"]["global_audit"])
            self.assertLess((root / "snapshot/raw.sqlite").stat().st_size, 100000)
            data.db.close()
            restored.db.close()


if __name__ == "__main__":
    unittest.main()
