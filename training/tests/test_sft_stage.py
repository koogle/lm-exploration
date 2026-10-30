"""Offline checks of SFT stages (`workflows/stages`); never imports Modal apps."""

import time
import unittest

from trainer.sft import Settings, train
from workflows.checks import parse_checkpoints
from workflows.stages import validate as validate_recipe

PARENT = "parent-run/step-000000100.pt"


def recipe(**changes):
    value = {
        "stage": "sft",
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
            {"stage": "rl"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(recipe(**changes))

    def test_extension_recipe(self):
        spec = validate_recipe(recipe(resume="sft-fixture-r01/step-000000200.pt", extend=True))
        self.assertTrue(spec["extend"])

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
