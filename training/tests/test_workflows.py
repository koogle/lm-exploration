"""Workflow selection, ownership and reproducible configuration resolution."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from workflows.checks import check_capacity, check_progress, check_resume
from workflows.config import load_workflow, resolve_payload

ROOT = Path(__file__).resolve().parents[1] / "workflows"


class WorkflowResolutionTests(unittest.TestCase):
    def test_all_shipped_workflows_resolve(self):
        for path in ROOT.glob("*.json"):
            with self.subTest(workflow=path.name):
                resolve_payload(load_workflow(path))

    def test_different_model_and_dataset_can_be_selected_without_controller_changes(self):
        bundle = load_workflow(
            ROOT / "h100x3-20b.json",
            model_config="model_configs/modern-124m.json",
            dataset_config="datasets/pilot-tokens.json",
        )
        _, dataset, training = resolve_payload(bundle)
        self.assertEqual((training.n_layer, training.n_embd), (14, 768))
        self.assertEqual(training.data_loader, "token_shards")
        self.assertEqual(training.target_tokens, bundle["workflow"]["target_tokens"])
        self.assertEqual(training.world_size, bundle["workflow"]["world_size"])
        with self.assertRaises(ValueError):
            load_workflow(ROOT / "h100x3-20b.json", model_config="../model.json")
        self.assertEqual(dataset.mode, "snapshot")
        self.assertNotIn("preparation_recipe", bundle)

    def test_conflicting_model_fields_and_incompatible_batches_fail_locally(self):
        for field in (
            "world_size",
            "target_tokens",
            "pilot",
            "allow_unreviewed_data",
            "data_loader",
        ):
            bundle = load_workflow(ROOT / "h100x3-20b.json")
            bundle["model"][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate"):
                resolve_payload(bundle)
        bundle = load_workflow(ROOT / "h100x3-20b.json")
        bundle["model"]["microbatch"] = 7
        with self.assertRaisesRegex(ValueError, "microbatches"):
            resolve_payload(bundle)

    def test_overlapped_preparation_persists_within_the_stall_timeout(self):
        bundle = load_workflow(ROOT / "climbmix-124m-overlap-smoke.json")
        bundle["workflow"]["persist_seconds"] = 3600
        with self.assertRaisesRegex(ValueError, "stall"):
            resolve_payload(bundle)
        bundle["workflow"]["overlap_preparation"] = False
        resolve_payload(bundle)

    def test_payload_is_frozen_when_source_files_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = load_workflow(ROOT / "h100x1-pilot.json")
            for key, filename in (
                ("workflow", "run.json"),
                ("model", "model.json"),
                ("dataset", "data.json"),
            ):
                value = copy.deepcopy(bundle[key])
                if key == "workflow":
                    value.update(model_config="model.json", dataset_config="data.json")
                (root / filename).write_text(json.dumps(value))
            resolved = load_workflow(root / "run.json")
            (root / "model.json").write_text("{}")
            self.assertEqual(resolve_payload(resolved)[2].n_layer, 14)
            with self.assertRaisesRegex(ValueError, "Warmup"):
                load_workflow(root / "run.json")


class WorkflowProgressTests(unittest.TestCase):
    def test_capacity_is_still_required(self):
        report = {
            "supported_mixture_exposures": 100,
            "sources": {"a": {"validation_loss_tokens": 1}},
        }
        check_capacity(report, 100)
        with self.assertRaises(ValueError):
            check_capacity(report, 101)
        report["sources"]["a"]["validation_loss_tokens"] = 0
        with self.assertRaises(ValueError):
            check_capacity(report, 100)

    def test_stops_and_resume_are_checked(self):
        report = {
            "total_loss_tokens": 95,
            "stop_reason": "token_limit",
            "final_validation": {"weighted_loss": 1.0},
        }
        self.assertTrue(check_progress(report, 80, 100, 10))
        for change in (
            {"total_loss_tokens": 80},
            {"total_loss_tokens": 101},
            {"stop_reason": "source_two_pass_limit"},
            {"final_validation": {"weighted_loss": float("nan")}},
        ):
            with self.assertRaises(ValueError):
                check_progress({**report, **change}, 80, 100, 10)
        # A session that only waited for prepared data continues the chain.
        waited = {**report, "total_loss_tokens": 80, "stop_reason": "data_wait"}
        self.assertFalse(check_progress(waited, 80, 100, 10))
        with self.assertRaises(ValueError):
            check_progress({**waited, "stop_reason": "time_limit"}, 80, 100, 10)
        check_resume(report, {"initial_validation": report["final_validation"]})
        with self.assertRaises(ValueError):
            check_resume(report, {"initial_validation": {"weighted_loss": 2.0}})
