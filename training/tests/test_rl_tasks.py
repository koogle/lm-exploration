import unittest

from training_data import rl_tasks
from training_data.code_execution import run_local


class LocalSandbox:
    def run(self, pairs):
        return run_local(pairs)


class RLTaskTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
