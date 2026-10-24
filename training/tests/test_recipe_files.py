import json
import tempfile
import unittest
from pathlib import Path

from training_data.simple_llm import load_recipe, resolve_files

LISTING = [
    "raw/20261019T000000Z/org--b-00000.jsonl.gz",
    "raw/20261019T000000Z/org--b-00001.jsonl.gz",
    "raw/20261020T000000Z/org--a-00000.jsonl.gz",
    "raw/20261020T000000Z/org--a-00001.jsonl.gz",
    "raw/20261020T000000Z/org--a-00002.jsonl.gz",
    "raw/20261021T000000Z/org--b-00000.jsonl.gz",
    "raw/20261021T000000Z/org--b-00001.jsonl.gz",
]


def listing(source, pattern):
    assert pattern == "raw/*/*.jsonl.gz"
    return list(reversed(LISTING))


def source(files):
    return {"name": "code", "repo": "r", "revision": "0" * 40, "files": files}


class RecipeFileTests(unittest.TestCase):
    def test_range_pattern(self):
        files = {"pattern": "shard_{:05d}.parquet", "range": [1, 12, 4]}
        self.assertEqual(
            resolve_files(source(files), listing),
            ["shard_00001.parquet", "shard_00005.parquet", "shard_00009.parquet"],
        )

    def test_glob_latest_export_interleaved(self):
        files = {"glob": "raw/*/*.jsonl.gz", "latest_export": True, "order": "interleave"}
        self.assertEqual(
            resolve_files(source(files), listing),
            [
                "raw/20261020T000000Z/org--a-00000.jsonl.gz",
                "raw/20261021T000000Z/org--b-00000.jsonl.gz",
                "raw/20261020T000000Z/org--a-00001.jsonl.gz",
                "raw/20261021T000000Z/org--b-00001.jsonl.gz",
                "raw/20261020T000000Z/org--a-00002.jsonl.gz",
            ],
        )

    def test_glob_defaults_to_sorted_paths(self):
        self.assertEqual(resolve_files(source({"glob": "raw/*/*.jsonl.gz"}), listing), LISTING)

    def test_lists_pass_through_and_bad_specs_fail(self):
        self.assertEqual(resolve_files(source(["a", "b"]), listing), ["a", "b"])
        for bad in (
            {"pattern": "x{}", "range": [0, 2]},
            {"glob": "raw/*/*.jsonl.gz", "order": "random"},
            {"glob": "raw/*/*.jsonl.gz", "shuffle": True},
            "shard.parquet",
        ):
            with self.assertRaises(ValueError):
                resolve_files(source(bad), listing)

    def test_load_recipe_expands_before_validation(self):
        recipe = {
            "id": "test",
            "sources": [
                {
                    **source({"glob": "raw/*/*.jsonl.gz", "latest_export": True}),
                    "format": "json.gz",
                    "weight_percent": 100,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            path.write_text(json.dumps(recipe))
            loaded = load_recipe(path, listing)
        self.assertEqual(len(loaded["sources"][0]["files"]), 5)


if __name__ == "__main__":
    unittest.main()
