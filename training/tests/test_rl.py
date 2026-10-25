import json
import random
import unittest
from pathlib import Path

from training_data import rl_tasks
from training_data.code_execution import run_local

from trainer.rl import Scheduler, Settings, advantages


class LocalSandbox:
    def run(self, pairs):
        return run_local(pairs)


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

    def test_rewards_need_eos_and_a_correct_answer(self):
        math = {"task": "math", "answer": "18"}
        code = {
            "task": "code",
            "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        }
        rows = [math, math, math, code, code]
        completions = [
            "16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\n#### 18",
            "So the answer is\n#### 17",
            "#### 18",
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a * b",
        ]
        finished = [True, True, False, True, True]
        self.assertEqual(
            rl_tasks.rewards(rows, completions, finished, LocalSandbox()), [1, 0, 0, 1, 0]
        )

    def test_prompts_match_sft_formats(self):
        self.assertEqual(
            rl_tasks.prompt({"task": "math", "question": "What is 2+2?"}),
            "Question: What is 2+2?\nAnswer:",
        )
        self.assertEqual(
            rl_tasks.prompt(
                {"task": "code", "question": "Add two numbers.", "declaration": "def add(a, b):"}
            ),
            "Question: Add two numbers.\nWrite a Python function `def add(a, b):`.\nAnswer:\n",
        )

    def test_recipes_validate(self):
        from modal_scripts.modal_rl_train import validate_recipe

        recipes = sorted(Path(__file__).parents[1].glob("rl/recipes/*.json"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=path.name):
                validate_recipe(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
