import json
import tempfile
import unittest
from pathlib import Path

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

    def test_contest_cases_run_like_livecodebench(self):
        stdin = rl_tasks.contest_tests([["1 2\n", "3\n"], ["5 7\n", "12"]], "")
        functional = rl_tasks.contest_tests(
            [["[1, 2]\n3", "[4, 5]"]], "class Solution:\n    def shift(self, xs, k):\n        "
        )
        pairs = [
            ("a, b = map(int, input().split())\nprint(a + b)", stdin, "contest"),
            (
                "import sys\na, b = map(int, sys.stdin.buffer.read().split())\nprint(a + b)",
                stdin,
                "contest",
            ),
            ("a, b = map(int, open(0).read().split())\nprint(a + b)\nexit(0)", stdin, "contest"),
            ("a, b = map(int, input().split())\nprint(a * b)", stdin, "contest"),
            ("import sys\nsys.exit(1)", stdin, "contest"),
            (
                "class Solution:\n    def shift(self, xs, k):\n        return [x + k for x in xs]",
                functional,
                "contest",
            ),
            (
                "class Solution:\n    def shift(self, xs, k):\n        return xs",
                functional,
                "contest",
            ),
            ("while True:\n    pass", stdin, "contest"),
        ]
        passed = [result["passed"] for result in run_local(pairs)]
        self.assertEqual(passed, [True, True, True, False, False, True, False, False])

    def test_thinking_rewards_read_the_answer_after_the_reasoning(self):
        row = {
            "task": "code",
            "answer_format": "thinking",
            "test_style": "contest",
            "tests": rl_tasks.contest_tests([["2\n", "4\n"]], ""),
        }
        right = "```python\nprint(int(input()) * 2)\n```"
        completions = [
            f"Draft:\n```python\nprint(0)\n```\n</think>\n\n{right}",
            f"I think\n{right}\nand so on",
            "Done.</think>\n```python\nprint(1)\n```",
            f"Done.</think>\n{right}",
        ]
        rewards = rl_tasks.rewards(
            [row] * 4, completions, [True, True, True, False], LocalSandbox(), unfinished=-0.5
        )
        self.assertEqual(rewards, [1, -0.5, 0, -0.5])

    def test_contest_rows_keep_the_example_and_a_sample_within_the_limits(self):
        tests = [{"input": f"{i}\n", "output": f"{i}\n"} for i in range(30)]
        tests[5] = {"input": "x" * (rl_tasks.CONTEST_CASE_CHARS + 1), "output": ""}
        cases = rl_tasks.contest_cases(tests, "seed")
        self.assertEqual(len(cases), rl_tasks.CONTEST_CASES)
        self.assertEqual(cases[0], ["0\n", "0\n"])
        self.assertNotIn(tests[5]["input"], [c[0] for c in cases])
        self.assertEqual(cases, rl_tasks.contest_cases(tests, "seed"))
        problem = {
            "question_id": "abc100_a",
            "question_content": "Print twice the number.",
            "starter_code": "",
            "tests": tests[:3],
            "platform": "atcoder",
            "difficulty": "easy",
            "contest_date": "2024-01-01T00:00:00",
        }
        row = rl_tasks.contest_row(problem, 0.0)
        self.assertEqual(
            (row["split"], row["answer_format"], row["cases"]), ("train", "thinking", "3 of 3")
        )
        self.assertIn("Print twice the number.", row["prompt"])
        self.assertIsNone(rl_tasks.contest_row({**problem, "tests": [tests[5]]}, 0.0))

    def test_mixed_pool_adds_function_problems_in_the_thinking_prompt(self):
        contest = [{"id": "contest-1", "task": "code", "split": "train", "test_style": "contest"}]
        functions = [
            {
                "id": f"kodcode-{index}",
                "task": "code",
                "split": "train",
                "question": f"Add two numbers, variant {index}.",
                "declaration": "def add(a, b):",
                "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                "test_style": "pytest",
            }
            for index in range(8)
        ] + [{"id": "gsm8k-1", "task": "math", "split": "train", "question": "1+1?"}]
        with tempfile.TemporaryDirectory() as folder:
            report = rl_tasks.build_mixed(contest, functions, 6, Path(folder) / "pool")
            rows = [
                json.loads(line)
                for line in (Path(folder) / "pool" / "pool.jsonl").read_text().splitlines()
            ]
        self.assertEqual(report["kept"], {"train/contest": 1, "train/pytest": 6})
        added = rows[1:]
        self.assertTrue(all(row["answer_format"] == "thinking" for row in added))
        starters = [row for row in added if "starter code" in row["prompt"]]
        named = [row for row in added if "Name it `add(a, b)`" in row["prompt"]]
        self.assertTrue(starters and named)
        self.assertEqual(len(starters) + len(named), 6)
        answer = "plan</think>```python\ndef add(a, b):\n    return a + b\n```"

        class Local:
            def run(self, pairs):
                return run_local(pairs)

        self.assertEqual(rl_tasks.rewards(added[:1], [answer], [True], Local()), [1.0])

    def test_contest_pool_keeps_problems_whose_teacher_program_passes(self):
        import tempfile

        def problem(qid, difficulty="easy", content="Echo the number."):
            return {
                "question_id": qid,
                "question_content": content,
                "starter_code": "",
                "tests": [{"input": "3\n", "output": "3\n"}],
                "platform": "atcoder",
                "difficulty": difficulty,
                "contest_date": "2024-01-01T00:00:00",
            }

        problems = [
            problem("ok"),
            problem("wrong"),
            problem("hard", "hard"),
            problem("leak", content="Leaked statement."),
            problem("untraced"),
        ]
        trace = "<think>\nhmm\n</think>\n```python\nprint(input())\n```"
        traces = [
            {"question_id": "ok", "completion": trace},
            {"question_id": "wrong", "completion": trace.replace("input()", "0")},
            {"question_id": "hard", "completion": trace},
            {"question_id": "leak", "completion": trace},
        ]
        with tempfile.TemporaryDirectory() as folder:
            report = rl_tasks.build_contest(
                problems,
                traces,
                lambda text: "Leaked" in text,
                run_local,
                Path(folder) / "pool",
                ("easy", "medium"),
                0.0,
            )
            pool = [json.loads(line) for line in (Path(folder) / "pool" / "pool.jsonl").open()]
        self.assertEqual([r["id"] for r in pool], ["contest-ok"])
        self.assertEqual(
            report["dropped"],
            {
                "problems": 5,
                "difficulty": 1,
                "contaminated": 1,
                "no_trace": 1,
                "reference_failed": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
