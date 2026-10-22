"""Offline checks of SFT recipe/argument validation; never imports Modal apps."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from trainer.repair_sft import Settings, train
from trainer.sft_recipe import load_recipe, parse_checkpoints, validate_recipe

PARENT = "parent-run/step-000000100.pt"


def recipe(**changes):
    value = {
        "name": "fixture",
        "run_id": "sft-fixture-r01",
        "parent": PARENT,
        "data": {"local": "artifacts/sft/repairs.jsonl"},
        "settings": {"steps": 10, "objective": "cig"},
    }
    value.update(changes)
    return value


class RecipeTests(unittest.TestCase):
    def test_defaults_and_normalized_settings(self):
        spec = validate_recipe(recipe())
        self.assertEqual(spec["gpu"], "L40S")
        self.assertEqual(spec["resume"], "")
        self.assertFalse(spec["extend"])
        self.assertEqual(spec["settings"]["steps"], 10)
        self.assertEqual(spec["settings"]["lr"], Settings().lr)
        Settings(**spec["settings"])

    def test_requires_explicit_parent_data_and_output(self):
        for field in ("run_id", "parent", "data", "settings"):
            value = recipe()
            del value[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_recipe(value)

    def test_rejects_invalid_values(self):
        for changes in (
            {"run_id": "../escape"},
            {"parent": "/runs/x/step-000000001.pt"},
            {"parent": "run/step-1.pt"},
            {"data": {"local": "a", "volume": "b"}},
            {"data": {"volume": "bad/id"}},
            {"gpu": "A100"},
            {"extend": True},
            {"extend": "yes", "resume": PARENT},
            {"settings": {"steps": 0}},
            {"settings": {"objective": "dpo"}},
            {"unexpected": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(recipe(**changes))

    def test_extension_recipe(self):
        spec = validate_recipe(recipe(resume="sft-fixture-r01/step-000000200.pt", extend=True))
        self.assertTrue(spec["extend"])

    def test_command_line_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            path.write_text(json.dumps(recipe()))
            spec = load_recipe(
                path, run_id="fresh-r02", parent="other/step-000000005.pt", data_id="data-v1"
            )
            self.assertEqual(spec["run_id"], "fresh-r02")
            self.assertEqual(spec["parent"], "other/step-000000005.pt")
            self.assertEqual(spec["data"], {"volume": "data-v1"})
            spec = load_recipe(path, dataset="other.jsonl")
            self.assertEqual(spec["data"], {"local": "other.jsonl"})
            self.assertEqual(spec["run_id"], "sft-fixture-r01")
            with self.assertRaisesRegex(ValueError, "either"):
                load_recipe(path, dataset="a.jsonl", data_id="b")

    def test_checkpoint_list(self):
        parsed = parse_checkpoints(f"base={PARENT}, sft=run/step-000000200.pt")
        self.assertEqual(list(parsed), ["base", "sft"])
        self.assertEqual(parsed["sft"], "run/step-000000200.pt")
        for value in ("", f"base={PARENT},base={PARENT}", "base", "base=/etc/passwd"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_checkpoints(value)


class DeadlineTests(unittest.TestCase):
    def test_train_refuses_after_deadline(self):
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            train("parent.pt", "data.jsonl", "out", Settings(), deadline=time.time() - 1)


if __name__ == "__main__":
    unittest.main()
