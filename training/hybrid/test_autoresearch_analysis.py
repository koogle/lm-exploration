"""Guard the research loop against incomparable checkpoints and premature winners."""

import unittest

from .autoresearch_analysis import paired, verdict


class ResearchDecisionTests(unittest.TestCase):
    def test_partial_improvement_is_not_a_winner(self):
        pairs = [{"delta": -0.02}, {"delta": -0.01}]
        self.assertEqual(verdict(pairs, False), "interim_only")
        self.assertEqual(verdict(pairs, True), "improved_proxy_ce")
        self.assertEqual(verdict(pairs[:1], True), "insufficient_paired_seeds")

    def test_opposing_seeds_are_inconclusive(self):
        self.assertEqual(verdict([{"delta": -0.1}, {"delta": 0.01}], True), "inconclusive_proxy_ce")

    def test_comparison_uses_equal_steps(self):
        def row(step, ce):
            return {
                "step": step,
                "tokens": step * 8192,
                "validation": {"ce": ce},
                "tokens_per_second": 1,
            }

        candidate = {"curve": [row(10, 4), row(20, 3)]}
        baseline = {"curve": [row(10, 4.1), row(30, 2)]}
        self.assertEqual(paired(candidate, baseline)["step"], 10)


if __name__ == "__main__":
    unittest.main()
