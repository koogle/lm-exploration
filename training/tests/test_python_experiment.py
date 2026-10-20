"""Corpus invariants that prevent silent mixture drift and held-out leakage."""

import tempfile
import unittest

from training_data.python_experiment import TARGETS, Corpus, category, split_for

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


if __name__ == "__main__":
    unittest.main()
