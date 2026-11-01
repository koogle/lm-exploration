"""Workflow plans (`workflows/plan.py`): stage order, parents, data, subsets and branches.
Resolves locally; never imports Modal apps."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainer.checkpoints import describe, parent
from workflows.plan import load_workflow, select
from workflows.stages import validate

ROOT = Path(__file__).resolve().parents[1]
START = "pretrain-run/step-000000100.pt"
RAW = "workflows/datasets/climbmix-code-v4-raw.json"
PRETRAIN = json.loads((ROOT / "workflows/h100x1-pilot.json").read_text())["stages"][-1]
SFT = {
    "stage": "sft",
    "data": {"volume": "math-data"},
    "settings": {"steps": 10, "task": "math"},
}

RL = {
    "stage": "rl",
    "pool": "rl-pool-fixture",
    "settings": {
        "algorithm": {"name": "off_policy/awr", "beta": 0.5},
        "trajectories": {"source": "file", "data": "teacher-trajectories"},
    },
}


def subword_pretrain(name="base", **changes):
    return {**PRETRAIN, "name": name, **changes}


def sft(name, **changes):
    return {**SFT, "name": name, **changes}


def plan_of(stages, family="subword", **fields):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder, "workflow.json")
        path.write_text(json.dumps({"family": family, "stages": stages, **fields}))
        return load_workflow(path, "flow", ROOT)


class WorkflowPlanTests(unittest.TestCase):
    def test_shipped_workflows_resolve(self):
        paths = sorted(ROOT.glob("workflows/*.json"))
        self.assertTrue(any("sft" in path.name for path in paths))
        for path in paths:
            with self.subTest(path=path.name):
                plan = load_workflow(path, "check")
                self.assertEqual(select(plan), [s["name"] for s in plan["stages"]])

    def test_full_workflow_chains_parents(self):
        plan = plan_of(
            [
                subword_pretrain(),
                subword_pretrain("long", stage="midtrain"),
                sft("chat"),
                {"stage": "sft", "branches": [sft("a"), sft("b")]},
            ]
        )
        parents = {s["name"]: s["parent"] for s in plan["stages"]}
        self.assertEqual(
            parents, {"base": None, "long": "base", "chat": "long", "a": "chat", "b": "chat"}
        )
        self.assertEqual(plan["stages"][2]["run_id"], "flow-chat")
        self.assertEqual(plan["stages"][2]["spec"]["settings"]["steps"], 10)
        self.assertNotIn("parent", plan["stages"][2]["spec"])

    def test_post_training_subset_starts_from_a_checkpoint(self):
        plan = plan_of([sft("chat"), sft("more")], **{"from": START})
        self.assertEqual([s["parent"] for s in plan["stages"]], ["from", "chat"])
        self.assertEqual(plan["from"], START)
        with self.assertRaisesRegex(ValueError, "from"):
            plan_of([sft("chat")])

    def test_data_stage_feeds_named_training_stage(self):
        data = {"stage": "data", "name": "corpus", "dataset_config": RAW, "version": "v1"}
        base = subword_pretrain(data="corpus")
        del base["dataset_config"]
        plan = plan_of([data, base])
        self.assertEqual(plan["stages"][1]["data"], "corpus")
        with self.assertRaisesRegex(ValueError, "unknown data stage"):
            plan_of([subword_pretrain(data="corpus")])

    def test_selection(self):
        plan = plan_of([subword_pretrain(), sft("a"), sft("b")])
        self.assertEqual(select(plan, "a"), ["a", "b"])
        self.assertEqual(select(plan, "", "a"), ["base", "a"])
        self.assertEqual(select(plan, "a", "a"), ["a"])
        for first, last in (("b", "a"), ("missing", "")):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                select(plan, first, last)

    def test_invalid_workflows(self):
        cases = (
            ([sft("a"), {"stage": "sft", "branches": [sft("b")]}, sft("c")], {}),  # last only
            ([sft("a"), sft("a")], {}),
            ([sft("a")], {"from": "/etc/passwd"}),
            ([sft("a")], {"extra": 1}),
            ([], {}),
            ([{"stage": "dpo", "name": "a"}], {}),
            ([sft("a", parent=START)], {}),
            ([sft("../a")], {}),
            ([subword_pretrain(), subword_pretrain("again")], {}),  # pretraining starts a model
            ([sft("a")], {"family": "hybrid"}),
        )
        for stages, fields in cases:
            fields = {"from": START, **fields}
            with self.subTest(stages=stages, fields=fields), self.assertRaises(ValueError):
                plan_of(stages, **fields)
        with self.assertRaises(ValueError):
            plan_of([sft("x" * 75)], **{"from": START})

    def test_command_prints_the_plan(self):
        printed = subprocess.run(
            [
                sys.executable,
                "-m",
                "workflows",
                "--workflow",
                "workflows/math-sft-5b.json",
                "--run-id",
                "check",
                "--start",
                "metamath-replay",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout
        plan = json.loads(printed)
        self.assertEqual(plan["selected"], ["metamath-replay"])
        self.assertTrue(plan["stages"][1]["spec"]["settings"]["replay"])


class RLStageTests(unittest.TestCase):
    def test_rl_stage_fills_settings(self):
        stage = {**RL, "run_id": "rl-fixture-r01", "parent": "sft-run/step-000000100.pt"}
        spec = validate(stage)
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
                validate({**stage, **changes})

    def test_shipped_rl_stages_use_stable_reinforce(self):
        plan = load_workflow(ROOT / "workflows/s0-rl-20b.json", "check")
        stages = [s for s in plan["stages"] if s["stage"] == "rl"]
        self.assertEqual([s["parent"] for s in stages], ["s0"] * 3)
        for stage in stages:
            self.assertEqual(stage["spec"]["settings"]["stable_reinforce"]["top_k"], 32)

    def test_hybrid_workflows_run_rl_but_not_sft(self):
        plan = plan_of([{**RL, "name": "rl"}], family="hybrid", **{"from": START})
        self.assertEqual([s["stage"] for s in plan["stages"]], ["rl"])
        with self.assertRaisesRegex(ValueError, "hybrid has no sft"):
            plan_of([sft("chat")], family="hybrid", **{"from": START})

    def test_workflow_runs_rl_after_sft(self):
        grpo = {**RL, "name": "grpo", "settings": {"algorithm": {"name": "on_policy/grpo"}}}
        plan = plan_of(
            [sft("chat"), {"stage": "rl", "branches": [{**RL, "name": "awr"}, grpo]}],
            **{"from": START},
        )
        self.assertEqual([s["stage"] for s in plan["stages"]], ["sft", "rl", "rl"])
        self.assertEqual([s["parent"] for s in plan["stages"]], ["from", "chat", "chat"])
        self.assertEqual(plan["stages"][1]["run_id"], "flow-awr")


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
                "stage": "rl",
                "architecture": {"module": "model_hybrid.model", "config": config},
                "identity": {"source_sha256": {"model": "m"}, "tokenizer_sha256": "t"},
            }
        )
        self.assertEqual(
            (tuned.stage, tuned.module, tuned.model_config, tuned.tokenizer_sha256),
            ("rl", "model_hybrid.model", config, "t"),
        )
        chat = describe(
            {
                "architecture": {"module": "model_hybrid.model", "config": config},
                "source_sha256": {"model": "m"},
                "data_identity": {"recipe": {"format": "chat-sft"}, "tokenizer_sha256": "t"},
            }
        )
        self.assertEqual(
            (pretrained.instructed, chat.stage, chat.instructed, tuned.instructed),
            (False, "pretrain", True, True),
        )
        with self.assertRaises(ValueError):
            describe({"model": {}})

    def test_parent_picks_its_family_and_checks_its_tokenizer(self):
        import torch
        from training_data.subword import TOKENIZER_SHA256

        saved = {
            "architecture": {"module": "model_100m.model", "config": {"n_layer": 2}},
            "source_sha256": {"model": "earlier model code"},
            "data_identity": {"tokenizer_sha256": TOKENIZER_SHA256},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "step-000000001.pt")
            torch.save(saved, path)
            _, description, family = parent(path)
            self.assertEqual(description.source_sha256, {"model": "earlier model code"})
            self.assertEqual(family.name, "subword")
            saved["architecture"]["module"] = "model_hybrid.model"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                parent(path)
            saved["architecture"]["module"] = "model_unknown.model"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "adapter"):
                parent(path)


if __name__ == "__main__":
    unittest.main()
