"""Workflow files of every stage and pipelines resolve locally; never imports Modal apps."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainer.checkpoints import describe
from workflows.config import load_workflow_config, workflow_stage
from workflows.pipeline import load_pipeline
from workflows.stages import load, validate

ROOT = Path(__file__).resolve().parents[1]
START = "pretrain-run/step-000000100.pt"
SFT = {
    "stage": "sft",
    "run_id": "recorded-r01",
    "parent": "other/step-000000001.pt",
    "data": {"volume": "math-data"},
    "settings": {"steps": 10, "task": "math"},
}

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


def pipeline(folder, steps, **changes):
    """A pipeline file in `folder` over SFT recipes named by `steps` (names or lists)."""
    for step in steps:
        for name in [step] if isinstance(step, str) else step:
            Path(folder, f"{name}.json").write_text(json.dumps(SFT))
    spec = {
        "stage": "pipeline",
        "from": START,
        "steps": [
            {"workflow": f"{s}.json"}
            if isinstance(s, str)
            else {"branches": [f"{n}.json" for n in s]}
            for s in steps
        ],
        **changes,
    }
    path = Path(folder, "pipeline.json")
    path.write_text(json.dumps(spec))
    return path


class WorkflowStageTests(unittest.TestCase):
    def test_every_workflow_file_names_its_stage(self):
        for path in [*ROOT.glob("workflows/*.json"), *ROOT.glob("sft/recipes/*.json")]:
            with self.subTest(path=path.name):
                self.assertIn(workflow_stage(path), ("pretrain", "sft"))
        pipelines = sorted(ROOT.glob("*/pipelines/*.json"))
        self.assertTrue(pipelines)
        for path in pipelines:
            with self.subTest(path=path.name):
                self.assertEqual(workflow_stage(path), "pipeline")
                load_pipeline(path, "pipeline-check")
        for path in ROOT.glob("workflows/*.json"):
            self.assertEqual(load_workflow_config(path).stage, "pretrain")

    def test_recipes_validate(self):
        for path in sorted(ROOT.glob("sft/recipes/*.json")):
            with self.subTest(recipe=path.name):
                load(path)
        recipes = sorted(ROOT.glob("rl/recipes/*.json"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=path.name):
                self.assertEqual(load(path)["settings"]["algorithm"]["score"], "centered")

    def test_pipeline_chains_stages_from_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = load_pipeline(pipeline(folder, ["sft-a", ["sft-b", "sft-c"]]), "pipe", folder)
        first, last = plan["steps"]
        self.assertEqual(first[0]["recipe"]["run_id"], "pipe-01-sft-a")
        self.assertEqual(first[0]["recipe"]["parent"], START)
        self.assertEqual([e["recipe"]["run_id"] for e in last], ["pipe-02-sft-b", "pipe-02-sft-c"])
        self.assertEqual(first[0]["recipe"]["settings"]["steps"], 10)
        self.assertEqual(
            set(plan["sources"]), {"pipeline", "pipe-01-sft-a", "pipe-02-sft-b", "pipe-02-sft-c"}
        )

    def test_invalid_pipelines(self):
        cases = (
            (["a", ["b", "c"], "d"], {}),  # only the last step branches
            (["a"], {"from": "/etc/passwd"}),
            ([["a", "a"]], {}),
            (["a"], {"extra": 1}),
            ([], {}),
        )
        for steps, changes in cases:
            with tempfile.TemporaryDirectory() as folder:
                path = pipeline(folder, steps, **changes)
                with self.subTest(steps=steps, changes=changes), self.assertRaises(ValueError):
                    load_pipeline(path, "pipe", folder)
        with tempfile.TemporaryDirectory() as folder:
            path = pipeline(folder, ["a"])
            with self.assertRaises(ValueError):
                load_pipeline(path, "x" * 75, folder)

    def test_command_prints_any_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pipeline(folder, ["sft-a"])
            Path(folder, "sft-a.json").write_text(json.dumps(SFT))
            printed = subprocess.run(
                [sys.executable, "-m", "workflows", "--workflow", str(Path(folder, "sft-a.json"))],
                capture_output=True,
                text=True,
                check=True,
                cwd=ROOT,
            ).stdout
            self.assertEqual(json.loads(printed)["settings"]["lr"], 1e-5)
            path.write_text(json.dumps({"stage": "dpo"}))
            with self.assertRaises(ValueError):
                workflow_stage(path)


class RLStageTests(unittest.TestCase):
    def test_rl_stage_fills_settings(self):
        spec = validate(RL)
        self.assertEqual(spec["settings"]["algorithm"]["max_weight"], 20.0)
        self.assertEqual(spec["settings"]["rollouts"]["samples_per_prompt"], 8)
        self.assertEqual(validate(spec), spec)
        for changes in (
            {"run_id": "../x"},
            {"pool": "a/b"},
            {"data": {"volume": "x"}},
            {"settings": {**RL["settings"], "trajectories": {"source": "file", "data": "../x"}}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate({**RL, **changes})

    def test_pipeline_runs_rl_after_sft(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "rl.json").write_text(json.dumps(RL))
            path = pipeline(folder, ["sft-a"])
            spec = json.loads(path.read_text())
            spec["steps"].append({"workflow": "rl.json"})
            path.write_text(json.dumps(spec))
            plan = load_pipeline(path, "pipe", folder)
        self.assertEqual([s[0]["stage"] for s in plan["steps"]], ["sft", "rl"])
        self.assertEqual(plan["steps"][1][0]["recipe"]["run_id"], "pipe-02-rl")


class CheckpointTests(unittest.TestCase):
    def test_describe_reads_both_layouts(self):
        config = {"n_layer": 2}
        pretrained = describe(
            {
                "architecture": {"module": "model_100m.model", "config": config},
                "source_sha256": {"model": "m"},
                "data_identity": {"tokenizer_sha256": "t"},
            }
        )
        self.assertEqual((pretrained.stage, pretrained.model_config), ("pretrain", config))
        tuned = describe(
            {
                "stage": "sft",
                "model_config": config,
                "identity": {"source_sha256": {"model": "m"}, "tokenizer_sha256": "t"},
            }
        )
        self.assertEqual(
            (tuned.stage, tuned.module, tuned.tokenizer_sha256), ("sft", "model_100m.model", "t")
        )
        with self.assertRaises(ValueError):
            describe({"model": {}})


if __name__ == "__main__":
    unittest.main()
