import unittest

from structured_tokenizer_report import paired


class StructuredReportTests(unittest.TestCase):
    def test_paired_savings_are_calculated_from_token_counts(self):
        old = {
            "tokens": 150,
            "per_parent": [
                {"parent_sha256": "a", "bytes": 100, "tokens": 50},
                {"parent_sha256": "b", "bytes": 200, "tokens": 100},
            ],
        }
        new = {
            "tokens": 120,
            "per_parent": [
                {"parent_sha256": "a", "bytes": 100, "tokens": 40},
                {"parent_sha256": "b", "bytes": 200, "tokens": 80},
            ],
        }
        result = paired(old, new)
        self.assertAlmostEqual(result["token_saving_percent"], 20)
        for bound in result["paired_ci95"]:
            self.assertAlmostEqual(bound, 20)

    def test_different_parent_text_is_rejected(self):
        old = {"tokens": 50, "per_parent": [{"parent_sha256": "a", "bytes": 100, "tokens": 50}]}
        new = {"tokens": 40, "per_parent": [{"parent_sha256": "a", "bytes": 101, "tokens": 40}]}
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired(old, new)
