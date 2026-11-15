"""Contest SFT conversations: harness prompts and programs verified by the harness grader."""

import unittest

from training_data import contest_sft

CODECONTESTS_ROW = {
    "description": "Print the sum of two integers.",
    "public_tests": {"input": ["2 3\n"], "output": ["5\n"]},
    "private_tests": {"input": ["10 -4\n"], "output": ["6\n"]},
    "generated_tests": {"input": [], "output": []},
    "solutions": {
        "language": [2, 3, 3],
        "solution": [
            "int main(){}",
            "a, b = map(int, input().split())\nprint(a + b + 1)",
            "a, b = map(int, input().split())\nprint(a + b)",
        ],
    },
    "input_file": "",
    "output_file": "",
}
LEETCODE_ROW = {
    "estimated_date": "2020-01-01",
    "problem_description": "Return the number of vowels in s.",
    "starter_code": "class Solution:\n    def countVowels(self, s: str) -> int:\n        ",
    "completion": (
        "class Solution:\n    def countVowels(self, s: str) -> int:\n"
        "        return len(list(filter(lambda c: c in ascii_lowercase[:1] + 'eiou', s)))"
    ),
    "input_output": "[{'input': 's = \"banana\"', 'output': '3'}, {'input': 's = \"xyz\"', 'output': '0'}]",
}


class ContestSFTTests(unittest.TestCase):
    def test_codecontests_keeps_a_passing_python3_program_in_the_harness_prompt(self):
        conversation = contest_sft.codecontests(CODECONTESTS_ROW)
        user, assistant = conversation["messages"]
        self.assertIn("Print the sum of two integers.", user["content"])
        self.assertIn("Read the inputs from stdin", user["content"])
        self.assertEqual(
            assistant["content"], "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        )

    def test_codecontests_without_a_passing_program_is_dropped(self):
        row = dict(CODECONTESTS_ROW, solutions={"language": [3], "solution": ["print(0)"]})
        self.assertIsNone(contest_sft.codecontests(row))

    def test_leetcode_cases_become_functional_tests(self):
        self.assertEqual(
            contest_sft.leetcode_tests(LEETCODE_ROW),
            [{"input": '"banana"', "output": "3"}, {"input": '"xyz"', "output": "0"}],
        )
        row = dict(LEETCODE_ROW, input_output="[{'input': 's = \"ab\"', 'output': 'ba'}]")
        self.assertEqual(contest_sft.leetcode_tests(row), [{"input": '"ab"', "output": '"ba"'}])

    def test_leetcode_writes_out_the_imports_a_solution_needs(self):
        program = contest_sft.leetcode(LEETCODE_ROW)["messages"][1]["content"]
        self.assertTrue(program.startswith("```python\nfrom typing import *"))
        self.assertIn("class Solution", program)

    def test_leetcode_drops_problems_from_the_cutoff_on(self):
        self.assertIsNone(contest_sft.leetcode(dict(LEETCODE_ROW, estimated_date="2025-02-01")))


if __name__ == "__main__":
    unittest.main()
