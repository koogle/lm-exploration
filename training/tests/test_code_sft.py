import json
import tempfile
import unittest
from pathlib import Path

from training_data.code_execution import run_local
from training_data.code_sft import (
    CODE,
    assert_tests,
    declaration,
    kodcode_row,
    mbpp_row,
    usable_tests,
)
from training_data.mixture_sft import MIXTURE, combine
from training_data.subword import LOSS_IGNORE_INDEX, load_frozen

KODCODE = {
    "style": "instruct",
    "subset": "Leetcode",
    "question_id": "Leetcode_3_I",
    "question": "Return the larger of two numbers.",
    "solution": 'def larger(a, b):\n    """Larger."""\n    return max(a, b)\n',
    "test": "from solution import larger\n\ndef test_larger():\n    assert larger(1, 2) == 2\n",
    "test_info": [{"function_name": "larger"}],
}
MBPP = {
    "task_id": 7,
    "text": "Write a function to add two numbers.",
    "code": "def add(x, y):\r\n  return x + y",
    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    "test_setup_code": "",
}


class CodeSFTTests(unittest.TestCase):
    def test_rows_keep_declarations_and_runnable_tests(self):
        kod, mbpp = kodcode_row(KODCODE, "train"), mbpp_row(MBPP, "train")
        self.assertEqual(kod["declaration"], "def larger(a, b):")
        self.assertEqual(kod["group"], "kodcode-Leetcode_3")
        self.assertEqual(mbpp["declaration"], "def add(x, y):")
        results = run_local([(r["solution"], r["tests"]) for r in (kod, mbpp)])
        self.assertTrue(all(r["passed"] for r in results))
        self.assertIsNone(kodcode_row({**KODCODE, "style": "complete"}, "train"))

    def test_fixture_and_parametrized_tests_are_unusable(self):
        self.assertTrue(usable_tests("def test_a():\n    assert True\n"))
        self.assertFalse(usable_tests("def test_a(tmp_path):\n    pass\n"))
        self.assertFalse(usable_tests("@p\ndef test_a():\n    pass\n"))
        self.assertFalse(usable_tests("x = 1\n"))
        self.assertIsNone(declaration("def f(): pass", "g"))
        self.assertIn("def test_asserts():\n    assert f(1)\n", assert_tests(["assert f(1)"]))

    def test_mixture_encodes_each_task_with_its_own_prompt(self):
        math = {
            "id": "m1",
            "group": "g1",
            "split": "train",
            "source": "s",
            "license": "MIT",
            "bucket": "gsm8k",
            "question": "What is 1+1?",
            "answer": "1+1=2\n#### 2",
        }
        code = {**kodcode_row(KODCODE, "validation")}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name, row in (("math", math), ("code", code)):
                (folder / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
            report = combine(
                {"math": folder / "math.jsonl", "code": folder / "code.jsonl"}, folder / "out"
            )
            self.assertEqual(report["rows"], {"code": 1, "math": 1})
            tokenizer = load_frozen()
            splits, _ = MIXTURE.prepare(folder / "out" / "examples.jsonl", tokenizer, 512)
        self.assertEqual(splits["train"][0]["bucket"], "math/gsm8k")
        self.assertEqual(splits["validation"][0]["bucket"], "code/kodcode/leetcode")
        encoded = splits["validation"][0]
        answer = [t for t in encoded["y"] if t != LOSS_IGNORE_INDEX]
        self.assertEqual(tokenizer.decode(answer[:-1]), CODE.answer(code))


if __name__ == "__main__":
    unittest.main()
