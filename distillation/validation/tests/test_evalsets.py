import json
import unittest
import unittest.mock

import helpers  # noqa: F401  (sets sys.path for the project modules below)

# isort: split
import evalsets
import grader


class EvalSetTest(unittest.TestCase):
    def test_code_string_parsing(self):
        self.assertEqual(
            evalsets.parse_code_string(
                'def solve(x):\\n    return \\"a\\"",\n  "explanation"'
            ),
            'def solve(x):\n    return "a"',
        )
        self.assertIsNone(evalsets.parse_code_string("def solve(x):\\n    return 1"))

    def test_student_prompt_round_trips_through_json(self):
        prompt = (
            evalsets.student_prompt('Say "hi"')
            + 'def solve():\\n    pass", "explanation": "x"}'
        )
        self.assertEqual(json.loads(prompt)["code"], "def solve():\n    pass")

    def test_mbpp_items_rename_tested_function(self):
        row = {
            "task_id": 2,
            "prompt": "Write a function to find the shared elements.",
            "code": "def helper(x):\n    return x\n\ndef similar_elements(a, b):\n    return tuple(set(a) & set(b))",
            "test_imports": [],
            "test_list": ["assert set(similar_elements((3, 4), (5, 4))) == set((4,))"],
        }
        item = evalsets.mbpp_items([row])[0]
        self.assertIn("`def solve(a, b):`", item["problem"])
        self.assertEqual(
            item["tests"], ["assert set(solve((3, 4), (5, 4))) == set((4,))"]
        )
        result = grader.run_case(
            {"key": "k", "code": item["reference"], "setup": "", "tests": item["tests"]}
        )
        self.assertTrue(result["passed"], result)

    def test_valid_tests_keeps_single_line_asserts(self):
        tests = [
            "assert solve(1) == 2",
            "x = 1",
            "assert other(1)",
            "assert solve(\n1)",
            3,
        ]
        self.assertEqual(evalsets.valid_tests(tests), ["assert solve(1) == 2"])

    def test_teacher_json_answer_with_fence(self):
        text = 'Sure\n```json\n{"code": "def solve():\\n    return 1"}\n```'
        self.assertEqual(
            evalsets.parse_json_answer(text, "code"), "def solve():\n    return 1"
        )


class GraderTest(unittest.TestCase):
    def test_pass_fail_error_and_timeout(self):
        cases = [
            {
                "key": "pass",
                "code": "def solve(x):\n    return x + 1",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "fail",
                "code": "def solve(x):\n    return x",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "error",
                "code": "def solve(x):\n    return y",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "exit",
                "code": "import sys\nsys.exit(0)",
                "setup": "",
                "tests": [],
            },
            {"key": "loop", "code": "while True:\n    pass", "setup": "", "tests": []},
        ]
        results = grader.grade(cases, timeout=2)
        self.assertTrue(results["pass"]["passed"])
        for key in ("fail", "error", "exit", "loop"):
            self.assertFalse(results[key]["passed"], key)
        self.assertIn("NameError", results["error"]["error"])


if __name__ == "__main__":
    unittest.main()
