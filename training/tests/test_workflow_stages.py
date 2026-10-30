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

ROOT = Path(__file__).resolve().parents[1]
START = "pretrain-run/step-000000100.pt"
RAW = "workflows/datasets/climbmix-code-v4-raw.json"
PRETRAIN = json.loads((ROOT / "workflows/h100x1-pilot.json").read_text())["stages"][-1]
SFT = {
    "stage": "sft",
    "data": {"volume": "math-data"},
    "settings": {"steps": 10, "task": "math"},
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

    def test_parent_from_other_model_code_needs_family_and_tokenizer(self):
        import model_100m.model as module
        import torch

        saved = {
            "architecture": {"module": module.__name__, "config": {"n_layer": 2}},
            "source_sha256": {"model": "earlier model code"},
            "data_identity": {"tokenizer_sha256": "t"},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "step-000000001.pt")
            torch.save(saved, path)
            _, description = parent(path, module, "t")
            self.assertEqual(description.source_sha256, {"model": "earlier model code"})
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                parent(path, module, "other")
            saved["architecture"]["module"] = "model_hybrid.model"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "family"):
                parent(path, module, "t")


if __name__ == "__main__":
    unittest.main()
