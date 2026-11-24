"""LiveCodeBench grading: private-test decoding, code extraction and the test modes."""

import base64
import json
import pickle
import unittest
import zlib

from evaluation.benchmarks import function_problems, livecodebench

STDIN = {
    "starter_code": "",
    "tests": [{"input": "2 3\n", "output": "5\n"}, {"input": "10 -4\n", "output": "6"}],
}
FUNCTIONAL = {
    "starter_code": "class Solution:\n    def addTwo(self, nums: List[int], k: int) -> List[int]:\n        ",
    "tests": [{"input": "[1, 2]\n3", "output": "[4, 5]"}],
}


def fenced(code):
    return f"Here it is.\n```python\n{code}\n```\nDone."


class LiveCodeBenchTests(unittest.TestCase):
    def test_private_tests_decode_and_refuse_classes(self):
        tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests))))
        self.assertEqual(livecodebench.private_tests(encoded), tests)
        hostile = base64.b64encode(zlib.compress(pickle.dumps(ValueError("x"))))
        with self.assertRaises(pickle.UnpicklingError):
            livecodebench.private_tests(hostile)

    def test_extracts_first_block_or_unclosed_block(self):
        self.assertEqual(livecodebench.extract_code(fenced("x = 1")), "x = 1\n")
        self.assertEqual(livecodebench.extract_code("```python\nx = 2\ny"), "x = 2\ny")
        self.assertIsNone(livecodebench.extract_code("no code"))
        self.assertFalse(livecodebench.finished("```python\nx = 2"))
        self.assertTrue(livecodebench.finished(fenced("x = 2")))

    def test_final_answer_follows_the_reasoning_span(self):
        answer = "```python\nprint(5)\n```"
        self.assertEqual(
            livecodebench.final_answer(f"<think>```python\nprint(4)\n```</think>\n{answer}"),
            f"\n{answer}",
        )
        self.assertEqual(livecodebench.final_answer(answer), answer)
        self.assertIsNone(livecodebench.final_answer("<think>still reasoning"))
        draft = "```python\nprint(4)\n```\nBetter:\n"
        self.assertEqual(livecodebench.last_block(draft + answer), answer)

    def test_stdin_programs_pass_only_on_every_test(self):
        right = "a, b = map(int, input().split())\nprint(a + b)"
        self.assertEqual(livecodebench.grade(STDIN, fenced(right)), 1.0)
        self.assertEqual(livecodebench.grade(STDIN, fenced("print(5)")), 0.0)
        self.assertEqual(livecodebench.grade(STDIN, fenced("raise SystemExit(1)")), 0.0)

    def test_functional_programs_call_the_starter_method(self):
        right = (
            "class Solution:\n    def addTwo(self, nums: List[int], k: int) -> List[int]:\n"
            "        return [n + k for n in nums]"
        )
        wrong = right.replace("n + k", "n")
        self.assertEqual(livecodebench.grade(FUNCTIONAL, fenced(right)), 1.0)
        self.assertEqual(livecodebench.grade(FUNCTIONAL, fenced(wrong)), 0.0)

    def test_check_problems_run_their_check_program_after_the_answer(self):
        humaneval = function_problems.humaneval(
            {
                "task_id": "HumanEval/0",
                "prompt": 'import math\n\ndef root(x):\n    """The square root of x."""\n',
                "test": "def check(candidate):\n    assert candidate(9) == 3",
                "entry_point": "root",
            }
        )
        whole = "def root(x):\n    return math.sqrt(x)"
        self.assertEqual(livecodebench.grade(humaneval, fenced(whole)), 1.0)
        self.assertEqual(livecodebench.grade(humaneval, fenced("    return math.sqrt(x)")), 1.0)
        self.assertEqual(livecodebench.grade(humaneval, fenced("def root(x):\n    return x")), 0.0)
        self.assertIn("def root(x):", livecodebench.instruction(humaneval))
        mbpp = function_problems.mbpp(
            {
                "task_id": 11,
                "text": "Write a function to add two numbers.",
                "test_list": ["assert add(2, 3) == 5", "assert add(1, 1) == 2"],
                "test_setup_code": "",
            }
        )
        right = "def add(a, b):\n    return a + b"
        self.assertEqual(livecodebench.grade(mbpp, fenced(right)), 1.0)
        self.assertEqual(livecodebench.grade(mbpp, fenced(right.replace("+", "*"))), 0.0)
        self.assertIn("assert add(2, 3) == 5", livecodebench.instruction(mbpp))
        self.assertIn(livecodebench.FUNCTION_FORMAT, livecodebench.instruction(mbpp))

    def test_prompts_carry_the_question_and_its_format(self):
        problem = {"question_content": "Add two numbers.", "starter_code": ""}
        self.assertTrue(livecodebench.instruction(problem).startswith(livecodebench.SYSTEM))
        self.assertIn("Question: Add two numbers.", livecodebench.instruction(problem))
        self.assertTrue(livecodebench.base_prompt(problem).endswith("backticks)\n\n"))

    def test_summary_groups_by_difficulty_and_platform(self):
        problems = [
            {"difficulty": "easy", "platform": "atcoder"},
            {"difficulty": "hard", "platform": "atcoder"},
        ]
        out = livecodebench.summary(problems, [1.0, 0.0])
        self.assertEqual(out["pass@1"], 0.5)
        self.assertEqual(out["difficulty"], {"easy": 1.0, "hard": 0.0})
        self.assertEqual(out["platform"], {"atcoder": 0.5})

    def test_verbatim_loop_needs_three_repeats_at_the_end(self):
        block = "Wait, let me check the sum again. "
        text = "First I read the input. " * 1 + "Then " + block * 5
        period, onset = livecodebench.verbatim_loop(text)
        self.assertEqual(period, len(block))
        self.assertAlmostEqual(onset, text.index(block) / len(text), delta=2 / len(text))
        self.assertIsNone(livecodebench.verbatim_loop("Then " + block * 2))
        self.assertIsNone(livecodebench.verbatim_loop(block * 5 + "```python\nprint(1)\n```"))


if __name__ == "__main__":
    unittest.main()
