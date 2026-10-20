"""Corpus invariants that prevent silent mixture drift and held-out leakage."""

import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from training_data.overnight_corpus import SCHEMA
from training_data.python_experiment import TARGETS, Corpus, category, split_for, tree_files
from training_data.subword import TOKENIZER_SHA256, load_frozen

TEXT = '# Keep the implementation comment and original indentation.\ndef add(a, b):\n    """Add two numbers."""\n    return a + b\n'


class PythonCorpusTests(unittest.TestCase):
    def test_exclusive_categories(self):
        self.assertEqual(category("src/add.py", TEXT)[0], "python_implementation")
        self.assertEqual(category("tests/test_add.py", TEXT)[0], "python_tests_examples")
        self.assertEqual(category("examples/add.py", TEXT)[0], "python_tests_examples")
        self.assertEqual(
            category("docs/tutorial.rst", "Python tutorial\n" + TEXT)[0], "python_documentation"
        )
        self.assertIsNone(category("src/add.rs", TEXT)[0])
        self.assertIsNone(category("vendor/add.py", TEXT)[0])
        self.assertEqual(sum(TARGETS.values()), 5_000_000_000)

    def test_malformed_explicit(self):
        self.assertEqual(category("x.py", TEXT + "\n???")[1], "invalid_python3")
        self.assertEqual(category("x.py", "# Automatically generated\n" + TEXT)[1], "generated")

    def test_dedup_heldout_wins_and_tokens_preserve_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            c = Corpus(folder)
            c.add(TEXT, "python_implementation", "owner/train", {}, split="train")
            row = c.db.execute("SELECT loss_tokens FROM documents").fetchone()
            self.assertEqual(
                row[0], len(c.tokenizer.encode(TEXT, add_special_tokens=False).ids) + 1
            )
            c.add(TEXT, "python_tests_examples", "owner/heldout", {}, split="validation")
            result = c.report("test")
            self.assertEqual(
                sum(v["unique_train_targets"] for v in result["categories"].values()), 0
            )
            self.assertFalse(result["capacity_sufficient"])
            c.db.close()

    def test_split_and_repository_cap(self):
        self.assertEqual(split_for("org/project"), split_for("org/project"))
        with tempfile.TemporaryDirectory() as folder:
            c = Corpus(folder)
            c.repo_counts["org/project", "python_implementation"] = (
                TARGETS["python_implementation"] // 100
            )
            c.add(TEXT, "python_implementation", "org/project", {}, split="train")
            self.assertEqual(c.report("test")["rejections"]["repository_one_percent_cap"], 1)
            c.db.close()

    def test_supplement_keeps_validation_and_excludes_cross_source_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "prior.sqlite"
            db = sqlite3.connect(source)
            db.executescript(SCHEMA)
            db.execute("CREATE TABLE settings (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute(
                "INSERT INTO settings VALUES(1,?)", (json.dumps({"tokenizer": TOKENIZER_SHA256}),)
            )
            tokenizer = load_frozen()
            for split, text in [
                ("train", TEXT),
                ("validation", TEXT.replace("return a + b", "return a - b")),
            ]:
                db.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
                    (
                        hashlib.sha256(text.encode()).hexdigest(),
                        "fineweb_edu",
                        split,
                        "example/" + split,
                        json.dumps({"revision": "pinned"}),
                        zlib.compress(text.encode()),
                        len(tokenizer.encode(text, add_special_tokens=False).ids) + 1,
                    ),
                )
            db.commit()
            db.close()
            c = Corpus(Path(folder) / "result")
            c.add(
                TEXT.replace("return a + b", "return a - b"),
                "python_implementation",
                "other/train",
                {},
                split="train",
            )
            result = c.supplement(source)
            self.assertEqual(
                result["categories"]["python_implementation"]["unique_train_targets"], 0
            )
            # A conflicting document is quarantined, not relabeled as validation.
            self.assertFalse(result["capacity_sufficient"])
            self.assertEqual(c.db.execute("SELECT COUNT(*) FROM excluded_hashes").fetchone()[0], 1)
            c.db.close()

    def test_batch_reader_preserves_committed_complete_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "--quiet", folder], check=True)
            (root / "example.py").write_text(TEXT)
            (root / ".gitattributes").write_text("example.py export-ignore\n")
            subprocess.run(["git", "-C", folder, "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    folder,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            blob = subprocess.check_output(
                ["git", "-C", folder, "rev-parse", "HEAD:example.py"], text=True
            ).strip()
            (root / "example.py").write_text("different uncommitted content")
            self.assertEqual(
                list(tree_files(root / ".git", [("example.py", blob)])), [("example.py", TEXT)]
            )


if __name__ == "__main__":
    unittest.main()
