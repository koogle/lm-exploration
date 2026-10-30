"""Workflow files of every stage resolve locally; never imports Modal apps."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainer.rl.recipe import validate_recipe
from workflows.config import STAGES, load_workflow_config, workflow_stage

ROOT = Path(__file__).resolve().parents[1]
RL = {
    "stage": "rl",
    "run_id": "rl-fixture-r01",
    "parent": "sft-run/step-000000100.pt",
    "pool": "rl-pool-fixture",
    "settings": {
        "algorithm": {"name": "off_policy/awr", "beta": 0.5},
        "trajectories": {"source": "file", "data": "teacher-trajectories"},
    },
}


class WorkflowStageTests(unittest.TestCase):
    def test_every_workflow_file_names_its_stage(self):
        files = [
            *ROOT.glob("workflows/*.json"),
            *ROOT.glob("sft/recipes/*.json"),
            *ROOT.glob("rl/recipes/*.json"),
        ]
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.name):
                self.assertIn(workflow_stage(path), STAGES)
        for path in ROOT.glob("workflows/*.json"):
            self.assertEqual(load_workflow_config(path).stage, "pretrain")

    def test_recipes_validate(self):
        from trainer.rl.recipe import load_recipe as load_rl
        from trainer.sft_recipe import load_recipe as load_sft

        for path in sorted(ROOT.glob("sft/recipes/*.json")):
            with self.subTest(recipe=path.name):
                load_sft(path)
        recipes = sorted(ROOT.glob("rl/recipes/*.json"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=path.name):
                self.assertEqual(load_rl(path)["settings"]["algorithm"]["score"], "centered")

    def test_rl_recipe_fills_settings(self):
        spec = validate_recipe(RL)
        self.assertEqual(spec["settings"]["algorithm"]["max_weight"], 20.0)
        self.assertEqual(spec["settings"]["rollouts"]["samples_per_prompt"], 8)
        for changes in (
            {"stage": "sft"},
            {"run_id": "../x"},
            {"pool": "a/b"},
            {"extra": 1},
            {"settings": {**RL["settings"], "trajectories": {"source": "file", "data": "../x"}}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe({**RL, **changes})

    def test_command_prints_any_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "rl.json")
            path.write_text(json.dumps(RL))
            printed = subprocess.run(
                [sys.executable, "-m", "workflows", "--workflow", str(path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=ROOT,
            ).stdout
            self.assertEqual(
                json.loads(printed)["settings"]["trajectories"]["answers_per_step"], 256
            )
            path.write_text(json.dumps({**RL, "stage": "dpo"}))
            with self.assertRaises(ValueError):
                workflow_stage(path)


if __name__ == "__main__":
    unittest.main()
