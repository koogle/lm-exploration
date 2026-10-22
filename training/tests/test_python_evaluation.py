import json
import sys
import tempfile
import unittest
from pathlib import Path

from sampling.recipe import load_recipe

from trainer.python_evaluation import (
    function_completion,
    grade,
    load_tasks,
    reference_cases,
    sample_cases,
    subprocess_executor,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "evaluation/python-tasks.json"


class PythonEvaluationTests(unittest.TestCase):
    def test_task_file_and_reference_cases(self):
        tasks = load_tasks(TASKS)
        self.assertEqual(len(tasks["tasks"]), 32)
        cases = reference_cases(tasks)
        self.assertEqual(cases[-1]["case_id"], "isolation_probe")
        self.assertEqual(len({c["case_id"] for c in cases}), 33)

    def test_completion_stops_at_next_top_level_statement(self):
        prompt = "def add(a, b):\n    "
        completion = "return a + b\n\n\nprint(add(1, 2))\ndef other():\n    pass\n"
        self.assertEqual(function_completion(prompt, completion), prompt + "return a + b\n\n")
        self.assertEqual(function_completion(prompt, "return a"), prompt + "return a")

    @unittest.skipUnless(sys.platform == "linux", "resource limits match the Linux sandbox")
    def test_reference_solutions_pass_and_wrong_or_spoofing_code_fails(self):
        cases = reference_cases(load_tasks(TASKS))[:4]
        broken = {**cases[0], "case_id": "broken", "code": "raise SystemExit(0)"}
        spoof = {
            **cases[1],
            "case_id": "spoof",
            "code": "import os\nos.write(1, b'x')\nraise SystemExit",
        }
        noisy = {**cases[2], "case_id": "noisy", "code": cases[2]["code"] + "\nprint('PASSED')"}
        results = {
            r["case_id"]: r for r in grade([*cases, broken, spoof, noisy], subprocess_executor)
        }
        for case in cases:
            self.assertTrue(results[case["case_id"]]["passed"], case["case_id"])
        self.assertFalse(results["broken"]["passed"])
        self.assertFalse(results["spoof"]["passed"])
        self.assertTrue(results["noisy"]["passed"])

    @unittest.skipUnless(sys.platform == "linux", "resource limits match the Linux sandbox")
    def test_sampling_results_are_graded_per_model_and_mode(self):
        tasks = load_tasks(TASKS)
        task = tasks["tasks"][0]
        samples = [
            {"prompt": task["prompt"], "completion": "return None\n", "mode": "greedy", "seed": 1},
            {"prompt": "Once upon a time", "completion": "...", "mode": "greedy", "seed": 1},
        ]
        result = {"models": [{"name": "tiny", "samples": samples}]}
        cases = sample_cases(tasks, result)
        self.assertEqual([c["task_id"] for c in cases], [task["task_id"]])
        rows = grade(cases, subprocess_executor)
        summary = summarize(rows)
        self.assertEqual(summary[0]["cases"], 1)
        self.assertEqual(summary[0]["pass_fraction"], 0.0)
        self.assertEqual(summary[0]["syntax_valid_fraction"], 1.0)

    def test_recipe_prompt_file_appends_task_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            recipe = {
                "name": "python-tasks",
                "precision": "fp32",
                "max_new_tokens": 256,
                "prompts": ["def add(a, b):\n    "],
                "prompt_file": str(TASKS),
                "prompt_group": "mbpp",
                "modes": [{"name": "greedy"}],
                "models": [
                    {"kind": "transformers", "name": "m", "repo": "r", "revision": "0" * 40}
                ],
            }
            path = Path(folder) / "recipe.json"
            path.write_text(json.dumps(recipe))
            loaded = load_recipe(path)
        self.assertEqual(len(loaded["prompts"]), 33)
        self.assertEqual(loaded["prompts"][1]["group"], "mbpp")


if __name__ == "__main__":
    unittest.main()
