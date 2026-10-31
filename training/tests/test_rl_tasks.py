import unittest

from training_data import rl_tasks
from training_data.sandbox import run_local


class LocalSandbox:
    def run(self, pairs):
        return run_local(pairs)


class RLTaskTests(unittest.TestCase):
    def test_rewards_need_eos_and_a_correct_answer(self):
        math = {"task": "math", "answer": "18", "answer_format": "plain"}
        code = {
            "task": "code",
            "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            "answer_format": "plain",
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

    def test_boxed_format_reads_the_last_box_and_code_block(self):
        self.assertEqual(
            rl_tasks.last_boxed("a \\boxed{1} then \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}"
        )
        self.assertIsNone(rl_tasks.last_boxed("no box \\boxed{open"))
        text = "First:\n```python\nx = 1\n```\nFinal:\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertEqual(rl_tasks.program(text, "boxed"), "def add(a, b):\n    return a + b\n")
        self.assertIsNone(rl_tasks.program("def add(a, b): ...", "boxed"))
        self.assertEqual(rl_tasks.program("def f(): ...", "plain"), "def f(): ...")

    def test_boxed_rewards(self):
        math = {"task": "math", "answer": "\\frac{1}{2}", "answer_format": "boxed"}
        code = {
            "task": "code",
            "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            "answer_format": "boxed",
        }
        rows = [math, math, code, code]
        completions = [
            "Half is \\boxed{0.5}.",
            "So \\boxed{2}.",
            "```python\ndef add(a, b):\n    return a + b\n```",
            "def add(a, b):\n    return a + b",
        ]
        rewards = rl_tasks.rewards(rows, completions, [True] * 4, LocalSandbox())
        self.assertEqual(rewards, [1, 0, 1, 0])

    def test_prompt_styles_set_the_prompt_and_answer_format(self):
        math = {"task": "math", "question": "What is 2+2? "}
        code = {"task": "code", "question": "Add two numbers.", "declaration": "def add(a, b):"}
        stack = {**code, "declaration": "class Stack:"}
        self.assertEqual(rl_tasks.styled(math, "sft")["answer_format"], "plain")
        self.assertEqual(rl_tasks.styled(math, "sft")["prompt"], rl_tasks.prompt(math))
        boxed = rl_tasks.styled(math, "boxed")
        self.assertEqual(boxed["answer_format"], "boxed")
        self.assertEqual(boxed["prompt"], f"What is 2+2?\n\n{rl_tasks.MATH_INSTRUCTION}")
        self.assertIn("Python function `add(a, b)`", rl_tasks.styled(code, "boxed")["prompt"])
        self.assertIn("Python class `Stack`", rl_tasks.styled(stack, "boxed")["prompt"])
        with self.assertRaisesRegex(ValueError, "prompt style"):
            rl_tasks.styled(math, "chat")

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
