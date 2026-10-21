"""Workflow selection, ownership and reproducible configuration resolution."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from workflows.config import DatasetConfig, load_workflow, resolve_payload

ROOT = Path(__file__).resolve().parents[1] / "workflows"


class WorkflowResolutionTests(unittest.TestCase):
    def test_all_shipped_workflows_resolve(self):
        for path in ROOT.glob("*.json"):
            with self.subTest(workflow=path.name):
                resolve_payload(load_workflow(path))

    def test_selected_model_dataset_and_execution_compose_one_request(self):
        bundle = load_workflow(ROOT / "small-5b.json")
        workflow, dataset, training = resolve_payload(bundle)
        self.assertEqual((training.n_layer, training.n_head, training.n_embd), (8, 3, 192))
        self.assertEqual(training.target_tokens, workflow.target_tokens)
        self.assertEqual(training.world_size, workflow.world_size)
        self.assertEqual(training.data_loader, dataset.loader)
        self.assertEqual(len(bundle["preparation_recipe"]["sources"][0]["revision"]), 40)
        self.assertIn("preparation_recipe", bundle["config_sources"])

    def test_different_model_and_dataset_can_be_selected_without_controller_changes(self):
        bundle = load_workflow(
            ROOT / "small-5b.json",
            model_config="model_configs/modern-124m.json",
            dataset_config="datasets/pilot-tokens.json",
        )
        _, dataset, training = resolve_payload(bundle)
        self.assertEqual((training.n_layer, training.n_embd), (14, 768))
        self.assertEqual(training.data_loader, "token_shards")
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
            bundle = load_workflow(ROOT / "small-5b.json")
            bundle["model"][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate"):
                resolve_payload(bundle)
        bundle = load_workflow(ROOT / "small-5b.json")
        bundle["model"]["microbatch"] = 7
        with self.assertRaisesRegex(ValueError, "microbatches"):
            resolve_payload(bundle)

    def test_no_5b_or_model_specific_ceiling(self):
        bundle = load_workflow(ROOT / "small-5b.json")
        bundle["workflow"].update(target_tokens=7_000_000_000, world_size=1)
        bundle["model"].update(n_layer=4, n_head=4, n_embd=256)
        self.assertEqual(resolve_payload(bundle)[2].target_tokens, 7_000_000_000)

    def test_dataset_modes_and_paths_are_explicit(self):
        for change in (
            {"root": "../escape"},
            {"mode": "unknown"},
            {"loader": "unknown"},
            {"recipe": "../escape.json"},
            {"loader": "token_shards"},
            {"snapshot": "v1/snapshots/run"},
        ):
            values = {
                "mode": "prepare_raw",
                "root": "corpus",
                "loader": "raw",
                "recipe": "data.json",
            }
            with self.subTest(change=change), self.assertRaises(ValueError):
                DatasetConfig(**{**values, **change})
        DatasetConfig(mode="snapshot", root="other", loader="raw", snapshot="v1/snapshots/run")

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

    def test_unknown_keys_and_escaping_config_references_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = json.loads((ROOT / "small-5b.json").read_text())
            for change in ({"model_config": "../model.json"}, {"typo": 1}):
                (root / "run.json").write_text(json.dumps({**values, **change}))
                with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                    load_workflow(root / "run.json")


if __name__ == "__main__":
    unittest.main()
