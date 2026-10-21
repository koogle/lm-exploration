import copy
import unittest
from dataclasses import asdict

from trainer.repair_sft import PILOT_TRAINER_SHA256, Settings, validate_resume_identity


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.old = {
            "settings": asdict(Settings()),
            "dataset_sha256": "data",
            "parent_sha256": "parent",
            "tokenizer_sha256": "tokens",
            "source_sha256": {"trainer": PILOT_TRAINER_SHA256, "cig": "cig", "model": "model"},
        }
        self.new = copy.deepcopy(self.old)
        self.new["settings"].update(steps=1000, save_every=1000, evaluate_every=200)
        self.new["source_sha256"]["trainer"] = "current-reviewed-source"

    def test_explicit_extension_and_strict_resume(self):
        validate_resume_identity(self.old, self.new, 200, extend=True)
        validate_resume_identity(self.new, self.new, 500)
        with self.assertRaises(ValueError):
            validate_resume_identity(self.old, self.new, 200)
        with self.assertRaises(ValueError):
            validate_resume_identity(self.new, self.new, 500, extend=True)

    def test_reject_learning_and_data_changes(self):
        for field, value in (
            ("lr", 2e-5),
            ("objective", "cig"),
            ("batch_size", 8),
            ("seed", 7),
            ("cig_proximity", 0.5),
            ("context", 2048),
        ):
            changed = copy.deepcopy(self.new)
            changed["settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_identity(self.old, changed, 200, extend=True)
        for field in ("dataset_sha256", "parent_sha256", "tokenizer_sha256"):
            changed = copy.deepcopy(self.new)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_identity(self.old, changed, 200, extend=True)
        for field in ("trainer", "model", "cig"):
            changed = copy.deepcopy(self.old)
            changed["source_sha256"][field] = "unreviewed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_identity(changed, self.new, 200, extend=True)


if __name__ == "__main__":
    unittest.main()
