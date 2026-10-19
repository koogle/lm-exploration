"""Family selection and paired loss aggregation have independent reference checks."""

import unittest

from trainer.eu_expanded_validation import offsets_for, paired_summary, select_families


class ExpandedValidationTests(unittest.TestCase):
    def test_selection_is_order_independent_and_parent_disjoint(self):
        rows = [(str(i), str(i // 3), 100) for i in range(60)]
        a, available, missing = select_families(rows, "test", limit=12)
        b, _, _ = select_families(list(reversed(rows)), "test", limit=12)
        self.assertEqual(a, b)
        self.assertEqual(available, 20)
        self.assertEqual(missing, 0)
        self.assertEqual(len({p for _, p, _ in a}), 12)
        self.assertNotEqual(a, select_families(rows, "other", limit=12)[0])

    def test_window_offsets_cover_only_real_targets(self):
        for count in (1, 2048, 2049, 10000):
            offsets = offsets_for("document", count)
            self.assertEqual(offsets, sorted(set(offsets)))
            self.assertLessEqual(len(offsets), 2)
            for start in offsets:
                self.assertEqual(start % 2048, 0)
                self.assertLess(start, count)
            self.assertEqual(offsets, offsets_for("document", count))
        # Long-document evaluation must not always select document prefixes.
        self.assertTrue(any(offsets_for(str(i), 100000)[0] > 0 for i in range(20)))

    def test_token_weighted_losses_and_clustered_paired_delta(self):
        rows = [
            {"window_id": "a1", "parent_id": "a", "category": "code", "tokens": 2, "nll_sum": 2.0},
            {"window_id": "a2", "parent_id": "a", "category": "code", "tokens": 2, "nll_sum": 2.0},
            {"window_id": "b1", "parent_id": "b", "category": "code", "tokens": 8, "nll_sum": 24.0},
        ]
        before = {"identity": {"sample_sha256": "same"}, "rows": rows}
        after = {
            "identity": {"sample_sha256": "same"},
            "rows": [{**r, "nll_sum": r["nll_sum"] + r["tokens"]} for r in rows],
        }
        result = paired_summary(before, after, {}, bootstrap_samples=100)["categories"]["code"]
        self.assertEqual(result["tokens"], 12)
        self.assertEqual(result["families"], 2)
        self.assertAlmostEqual(result["before_loss"], 28 / 12)
        self.assertAlmostEqual(result["after_loss"], 40 / 12)
        self.assertAlmostEqual(result["delta_loss"], 1)
        self.assertEqual(result["delta_ci95"], [1.0, 1.0])
        self.assertEqual(result["direction"], "worse")
        bad = {**after, "rows": [{**after["rows"][0], "tokens": 3}, *after["rows"][1:]]}
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            paired_summary(before, bad, {})
        with self.assertRaisesRegex(ValueError, "identical windows"):
            paired_summary(before, {**after, "rows": after["rows"][:-1]}, {})


if __name__ == "__main__":
    unittest.main()
