import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from trainer.continuation_generation import repetition_metrics
from trainer.python_evaluation import execution_script, reference_cases, summarize

TASKS = Path(__file__).resolve().parents[1] / "evaluation/python-tasks.json"


def run(case):
    marker = uuid.uuid4().hex
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", execution_script(case, marker)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout == marker and result.returncode == 0


class PythonEvaluationTests(unittest.TestCase):
    def test_fixed_task_file_shape(self):
        tasks = json.loads(TASKS.read_text())
        self.assertEqual(len(tasks["tasks"]), 32)
        cases = reference_cases(tasks)
        self.assertEqual(cases[-1]["task_id"], "isolation_probe")
        self.assertEqual(len({c["task_id"] for c in cases}), 33)

    @unittest.skipUnless(sys.platform == "linux", "resource limits match the Linux sandbox")
    def test_reference_solutions_pass_and_wrong_code_fails(self):
        cases = reference_cases(json.loads(TASKS.read_text()))[:4]
        for case in cases:
            with self.subTest(task=case["task_id"]):
                self.assertTrue(run(case))
        broken = {**cases[0], "code": "raise SystemExit(0)"}
        self.assertFalse(run(broken))
        noisy = {**cases[0], "code": cases[0]["code"] + "\nprint('spoof')"}
        self.assertTrue(run(noisy))

    def test_repetition_metrics_and_summary(self):
        looped = repetition_metrics([5, 6] * 40)
        self.assertTrue(looped["long_loop"])
        self.assertGreater(looped["repeated_token_4gram_fraction"], 0.9)
        varied = repetition_metrics(list(range(100)))
        self.assertFalse(varied["long_loop"])
        self.assertEqual(varied["repeated_token_4gram_fraction"], 0.0)
        sample = {"syntax_valid": True, "new_tokens": 80, "stop_reason": "eos", **looped}
        summary = summarize([sample, {**sample, "syntax_valid": False}])
        self.assertEqual(summary["syntax_valid_fraction"], 0.5)
        self.assertEqual(summary["long_loop_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
