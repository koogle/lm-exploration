import copy
import unittest

from megakernel.build_sweep import rank_trials, schedule, variants


class SweepTests(unittest.TestCase):
    def test_balanced_matched_schedule(self):
        plan = schedule()
        self.assertEqual(len(plan), 16)
        self.assertEqual([v for _, v in plan[:8]], [v for _, v in plan[8:]][::-1])
        for threads in (256, 512):
            for blocks in (66, 132):
                pair = [
                    v
                    for v in variants()
                    if v["threads"] == threads and v["blocks"] == blocks
                ]
                self.assertEqual(
                    {v["backend"] for v in pair}, {"manual", "thunderkittens"}
                )

    def test_incomplete_or_incorrect_results_cannot_win(self):
        rows = [
            {
                "variant": variants()[0],
                "round": r,
                "result": {
                    "status": "completed",
                    "correctness_passed": True,
                    "reset_reproducible": True,
                    "performance_eligible": True,
                    "measurements": [
                        {"backend": "megakernel", "accepted_examples_per_hour": speed}
                    ],
                },
            }
            for r, speed in enumerate((100, 120))
        ]
        self.assertEqual(rank_trials(rows)[0]["median_accepted_examples_per_hour"], 110)
        self.assertEqual(rank_trials(rows[:1]), [])
        self.assertEqual(rank_trials([rows[0], rows[0]]), [])
        for field in (
            "correctness_passed",
            "reset_reproducible",
            "performance_eligible",
        ):
            failed = copy.deepcopy(rows)
            failed[0]["result"][field] = False
            self.assertEqual(rank_trials(failed), [])


if __name__ == "__main__":
    unittest.main()
