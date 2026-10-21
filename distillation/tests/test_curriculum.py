import unittest

from distillation.curriculum import task, verify
from distillation.curriculum_train import make_windows


class CurriculumTests(unittest.TestCase):
    def test_specs_unique_and_transformations_held_out(self):
        tasks = [task(i) for i in range(512)]
        self.assertEqual(len({t["id"] for t in tasks}), 512)
        self.assertEqual(sum(t["split"] == "validation" for t in tasks), 128)
        for item in tasks:
            self.assertEqual(item["split"] == "validation", item["variant"] == 3)

    def test_factorial_not_square_and_no_input_mutation(self):
        correct = "def solve(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result % 488"
        self.assertTrue(verify(correct, task(0)["cases"])["passed"])
        self.assertFalse(
            verify("def solve(n):\n    return n*n", task(0)["cases"])["passed"]
        )
        self.assertFalse(
            verify(
                "def solve(items):\n    items.sort()\n    return items",
                [([[2, 1]], [1, 2])],
            )["passed"]
        )

    def test_reject_unsafe_code_and_bound_nontermination(self):
        for code in (
            "import os\ndef solve(n):\n    return n",
            "def solve(n):\n    return open('/etc/passwd').read()",
            "def solve(n):\n    return n.__class__",
            "def solve(n):\n    while True:\n        pass",
        ):
            with self.subTest(code=code):
                self.assertFalse(verify(code, [([1], 1)])["passed"])

    def test_chunk_targets_once_and_include_eos(self):
        class Tokenizer:
            def encode(self, text):
                class Encoding:
                    def __init__(self):
                        self.ids = list(range(4, 23))

                return Encoding()

        records = [
            {"prompt": "problem", "answer": {"code": "code", "explanation": "why"}}
        ]
        windows = make_windows(records, Tokenizer(), context=7)
        self.assertEqual([i for _, y in windows for i in y], list(range(4, 23)) + [2])
        for x, y in windows:
            self.assertEqual(x[1:], y[:-1])
        self.assertEqual(windows[0][0][0], 1)


if __name__ == "__main__":
    unittest.main()
