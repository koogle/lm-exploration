import random
import unittest

from trainer.rl import Scheduler, Settings, advantages


class RLTests(unittest.TestCase):
    def test_advantages_subtract_the_group_mean(self):
        for got, expected in zip(
            advantages([1, 0, 0, 1, 1, 1], 3), [2 / 3, -1 / 3, -1 / 3, 0, 0, 0], strict=True
        ):
            self.assertAlmostEqual(got, expected)

    def test_scheduler_rests_prompts_solved_by_all_or_none_twice(self):
        rows = [{"id": str(i), "task": "math"} for i in range(4)]
        scheduler = Scheduler(rows, rest=10, rng=random.Random(0))
        scheduler.update(rows[0], 0.0, 0)
        self.assertEqual(scheduler.resting(1), 0)
        scheduler.update(rows[0], 0.0, 1)
        scheduler.update(rows[1], 1.0, 1)
        scheduler.update(rows[1], 0.5, 2)
        self.assertEqual(scheduler.resting(2), 1)
        drawn = {r["id"] for _ in range(20) for r in scheduler.draw("math", 3, 2)}
        self.assertNotIn("0", drawn)
        self.assertEqual(scheduler.resting(11), 0)

    def test_settings(self):
        self.assertEqual(Settings(lr=1e-6, warmup_steps=10).learning_rate(4), 5e-7)
        self.assertEqual(Settings().token_limit("code"), 384)
        for values in ({"samples_per_prompt": 1}, {"replay_weight": 0.1}, {"math_share": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values)


if __name__ == "__main__":
    unittest.main()
