import json
import tempfile
import unittest
from pathlib import Path

from training_data.code_execution import run_local
from training_data.code_sft import (
    CODE,
    assert_tests,
    declaration,
    definitions,
    mbpp_row,
    opencodeinstruct_row,
)
from training_data.mixture_sft import MIXTURE, combine
from training_data.subword import LOSS_IGNORE_INDEX, load_frozen

OCI = {
    "id": "abc123",
    "domain": "generic",
    "input": "Return the larger of two numbers.",
    "output": "Here it is:\n```python\nimport math\n\n\ndef larger(a, b):\n"
    '    """Larger."""\n    return max(a, b)\n\n\n# Example usage:\nx = larger(1, 2)\n'
    "print(x)\n```\nIt returns the maximum.",
    "unit_tests": '["\\nassert larger(1, 2) == 2\\n", "\\nassert larger(\\n    5, 3\\n) == 5\\n"]',
    "tests_execution_status": '["pass", "pass"]',
}
MBPP = {
    "task_id": 7,
    "text": "Write a function to add two numbers.",
    "code": "def add(x, y):\r\n  return x + y",
    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    "test_setup_code": "",
}


class CodeSFTTests(unittest.TestCase):
    def test_rows_keep_definitions_and_runnable_tests(self):
        oci, mbpp = opencodeinstruct_row(OCI, "train"), mbpp_row(MBPP, "train")
        self.assertEqual(oci["declaration"], "def larger(a, b):")
        self.assertNotIn("Example", oci["solution"])
        self.assertTrue(oci["solution"].startswith("import math\n\ndef larger"))
        self.assertEqual(mbpp["declaration"], "def add(x, y):")
        results = run_local([(r["solution"], r["tests"]) for r in (oci, mbpp)])
        self.assertTrue(all(r["passed"] for r in results))
        failed = {**OCI, "tests_execution_status": '["pass", "fail"]'}
        self.assertIsNone(opencodeinstruct_row(failed, "train"))

    def test_declarations_and_definitions(self):
        self.assertIsNone(declaration("def f(): pass", "g"))
        self.assertEqual(declaration("class Car:\n    pass", "Car"), "class Car:")
        self.assertEqual(definitions("@d\ndef f():\n    pass\nf()\n"), "@d\ndef f():\n    pass")
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
        code = opencodeinstruct_row(OCI, "validation")
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
        self.assertEqual(splits["validation"][0]["bucket"], "code/oci/generic")
        encoded = splits["validation"][0]
        answer = [t for t in encoded["y"] if t != LOSS_IGNORE_INDEX]
        self.assertEqual(tokenizer.decode(answer[:-1]), CODE.answer(code))


if __name__ == "__main__":
    unittest.main()
