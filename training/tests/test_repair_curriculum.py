import tempfile
import unittest
from pathlib import Path

from training_data.repair_curriculum import LANGUAGES, execute, oracle, rows, write_dataset
from training_data.repair_sft import prepare
from training_data.subword import load_frozen


class CurriculumTests(unittest.TestCase):
    def test_family_split_and_real_tokenizer(self):
        records = [r for language in LANGUAGES for r in rows(language)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "repairs.jsonl"
            write_dataset(path, records)
            splits, _ = prepare(path, load_frozen(), 1024)
        self.assertEqual(len(splits["train"]), 576)
        self.assertEqual(len(splits["validation"]), 36)
        train = {r["family"] for r in records if r["split"] == "train"}
        heldout = {r["family"] for r in records if r["split"] == "validation"}
        self.assertFalse(train & heldout)
        self.assertTrue(all(r["cases"] for r in records))

    def test_oracle_known_answers(self):
        expected = {
            "sum": 2,
            "difference": 10,
            "product": -9,
            "weighted": 5,
            "squares": 28,
            "double_sum": 1,
            "clamp": 3,
            "threshold": 0,
            "absolute_difference": 10,
            "maximum": 6,
        }
        self.assertEqual({f: oracle(f, 3, -4, 3) for f in expected}, expected)

    def test_authored_python_and_shell_references(self):
        for language in ("python", "bash"):
            # One parameter per task family; all variants are verified remotely before training.
            seen = set()
            for row in rows(language):
                if row["family"] in seen:
                    continue
                seen.add(row["family"])
                for variant, expected in (("broken", False), ("fixed", True)):
                    with self.subTest(language=language, family=row["family"], variant=variant):
                        with tempfile.TemporaryDirectory() as folder:
                            result = execute(language, row[variant], row["cases"], folder)
                        self.assertEqual(result["passed"], expected)


if __name__ == "__main__":
    unittest.main()
