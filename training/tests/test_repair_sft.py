import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training_data.repair_sft import REPAIR

from trainer.sft import Settings


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) + 4 for c in text])


def example(name="one", split="train"):
    return {
        "id": name,
        "language": "python",
        "group": name,
        "split": split,
        "instruction": f"Return {name}.",
        "broken": f"def {name}(): return 0",
        "fixed": f"def {name}(): return 1",
        "source": "synthetic test fixture",
        "license": "CC0-1.0",
    }


class RepairTests(unittest.TestCase):
    def test_answer_alignment_and_eos(self):
        row = example()
        encoded = REPAIR.encode(row, Tokenizer(), 1024)
        prefix_len = 1 + len(REPAIR.prompt(row))
        self.assertEqual(encoded["y"][: prefix_len - 1], [-100] * (prefix_len - 1))
        self.assertEqual(encoded["y"][prefix_len - 1], ord("d") + 4)
        self.assertEqual(encoded["y"][-1], 2)
        self.assertEqual(len(encoded["x"]), len(encoded["y"]))
        self.assertEqual(sum(y != -100 for y in encoded["y"]), len(row["fixed"]) + 1)

    def test_real_frozen_tokenizer(self):
        from training_data.subword import load_frozen

        tokenizer = load_frozen()
        row = example()
        encoded = REPAIR.encode(row, tokenizer, 1024)
        labels = [v for v in encoded["y"] if v != -100]
        self.assertEqual(labels[-1], 2)
        self.assertEqual(tokenizer.decode(labels[:-1]), row["fixed"])

    def test_multiple_languages(self):
        from training_data.subword import load_frozen

        tokenizer = load_frozen()
        snippets = {
            "javascript": "function add(a,b) { return a + b; }",
            "typescript": "const add = (a: number, b: number): number => a + b;",
            "go": "func add(a, b int) int { return a + b }",
            "rust": "fn add(a: i32, b: i32) -> i32 { a + b }",
            "c": "int add(int a, int b) { return a + b; }",
            "cpp": "int add(int a, int b) { return a + b; }",
            "java": "static int add(int a, int b) { return a + b; }",
            "bash": 'printf "%s\\n" "$((a + b))"',
        }
        rows = []
        for lang, code in snippets.items():
            for split in ("train", "validation"):
                row = example(f"{lang}-{split}", split)
                comment = f" # {lang}-{split}" if lang == "bash" else f" /* {lang}-{split} */"
                row.update(
                    language=lang, fixed=code + comment, broken=code.replace("+", "-") + comment
                )
                encoded = REPAIR.encode(row, tokenizer, 1024)
                labels = [v for v in encoded["y"] if v != -100]
                self.assertEqual(tokenizer.decode(labels[:-1]), row["fixed"])
                self.assertIn(f"Repair the {lang} code", REPAIR.prompt(row))
                self.assertEqual(encoded["bucket"], lang)
                rows.append(row)
        splits, _ = self.prepare_rows(rows)
        self.assertEqual(len(splits["train"]), len(snippets))

    def test_missing_language_and_validation_coverage(self):
        row = example()
        del row["language"]
        with self.assertRaisesRegex(ValueError, "language"):
            REPAIR.encode(row, Tokenizer(), 1024)
        a, b = example(), example("two", "validation")
        a["language"] = "rust"
        with self.assertRaisesRegex(ValueError, "Missing validation languages"):
            self.prepare_rows([a, b])

    def test_no_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            REPAIR.encode(example(), Tokenizer(), 10)

    def prepare_rows(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            return REPAIR.prepare(path, Tokenizer(), 1024)

    def test_split_and_hash(self):
        splits, digest = self.prepare_rows([example(), example("two", "validation")])
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(splits["train"]), 1)

    def test_family_leak(self):
        a, b = example(), example("two", "validation")
        b["group"] = a["group"]
        with self.assertRaisesRegex(ValueError, "Group crosses"):
            self.prepare_rows([a, b])

    def test_solution_leak(self):
        a, b = example(), example("two", "validation")
        b["fixed"] = a["fixed"]
        with self.assertRaisesRegex(ValueError, "Duplicate text crosses"):
            self.prepare_rows([a, b])

    def test_invalid_data_and_settings(self):
        with self.assertRaisesRegex(ValueError, "Both train"):
            self.prepare_rows([example()])
        for values in ({"steps": 0}, {"lr": float("nan")}, {"max_seconds": 14401}):
            with self.assertRaises(ValueError):
                Settings(**values)


class TensorBatchTests(unittest.TestCase):
    def test_padding_does_not_become_supervision(self):
        from trainer.batches import collate

        rows = [{"x": [1, 5, 6], "y": [-100, 6, 2]}, {"x": [1], "y": [2]}]
        x, y = collate(rows, "cpu")
        self.assertEqual(x.tolist(), [[1, 5, 6], [1, 0, 0]])
        self.assertEqual(y.tolist(), [[-100, 6, 2], [2, -100, -100]])


if __name__ == "__main__":
    unittest.main()
