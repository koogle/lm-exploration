"""LiveCodeBench grading: private-test decoding, code extraction and both test modes."""

import base64
import json
import pickle
import unittest
import zlib

from model_hybrid import livecodebench

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

    def test_summary_groups_by_difficulty_and_platform(self):
        problems = [
            {"difficulty": "easy", "platform": "atcoder"},
            {"difficulty": "hard", "platform": "atcoder"},
        ]
        out = livecodebench.summary(problems, [1.0, 0.0])
        self.assertEqual(out["pass@1"], 0.5)
        self.assertEqual(out["difficulty"], {"easy": 1.0, "hard": 0.0})
        self.assertEqual(out["platform"], {"atcoder": 0.5})


if __name__ == "__main__":
    unittest.main()
