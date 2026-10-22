import json
import tempfile
import unittest
from pathlib import Path

from distillation.recipe import load_recipe, validate_recipe

TEACHER = {"repo": "Qwen/Qwen3-4B-Base", "revision": "0" * 40}
DATA = {"manifest": "snapshot/manifest.json", "database": "code/raw.sqlite"}


def shared(**changes):
    recipe = {
        "kind": "shared",
        "run_id": "shared-run",
        "teacher": TEACHER,
        "parent": "parent-run/step-000000010.pt",
        "data": DATA,
        "target_tokens": 1000,
        "max_seconds": 600,
    }
    recipe.update(changes)
    return recipe


class RecipeTests(unittest.TestCase):
    def test_valid_kinds_normalize_defaults(self):
        spec = validate_recipe(shared())
        self.assertEqual(spec["resume"], "")
        curriculum = validate_recipe(
            {"kind": "curriculum", "run_id": "c", "teacher": TEACHER, "count": 64}
        )
        self.assertEqual(curriculum["sampling"]["temperature"], 0.7)
        evaluation = validate_recipe(
            {
                "kind": "evaluate",
                "run_id": "e",
                "data": DATA,
                "checkpoints": ["training:p/step-000000001.pt"],
            }
        )
        self.assertEqual(evaluation["max_documents"], 512)
        generations = validate_recipe(
            {
                "kind": "generations",
                "run_id": "g",
                "checkpoints": ["run/student-1/step-000000002.pt"],
                "precision": "fp32",
                "max_new_tokens": 8,
                "prompts": ["def f():", {"text": "x =", "group": "g"}],
                "modes": [{"name": "greedy", "temperature": 0}],
            }
        )
        self.assertEqual(len(generations["checkpoints"]), 1)

    def test_rejects_implicit_or_unsafe_inputs(self):
        bad = [
            shared(parent=""),
            shared(parent="../escape/step-000000001.pt"),
            shared(data={"manifest": "../m.json", "database": "d"}),
            shared(teacher={"repo": "Qwen/Qwen3-4B-Base", "revision": "main"}),
            shared(max_seconds=90000),
            shared(target_tokens=0),
            shared(unexpected=True),
            {"kind": "students", "run_id": "s"},
            {"kind": "unknown", "run_id": "x"},
            {
                "kind": "evaluate",
                "run_id": "e",
                "data": DATA,
                "checkpoints": ["p/step-000000001.pt"],
            },
            {"kind": "curriculum", "run_id": "c", "teacher": TEACHER, "count": 8},
        ]
        for recipe in bad:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                validate_recipe(recipe)

    def test_byte_time_cap_and_overrides(self):
        with self.assertRaises(ValueError):
            validate_recipe(shared(kind="byte", max_seconds=1201))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            path.write_text(json.dumps(shared()))
            spec = load_recipe(path, "fresh", "", "old-run/step-000000500.pt")
            self.assertEqual(
                (spec["run_id"], spec["resume"]), ("fresh", "old-run/step-000000500.pt")
            )
            path.write_text(
                json.dumps({"kind": "students", "run_id": "s", "corpus": "c"})
            )
            with self.assertRaises(ValueError):
                load_recipe(path, parent="p/step-000000001.pt")


if __name__ == "__main__":
    unittest.main()
