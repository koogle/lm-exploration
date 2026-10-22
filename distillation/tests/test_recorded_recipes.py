import unittest
from pathlib import Path

from distillation.recipe import load_recipe


class RecordedRecipeTests(unittest.TestCase):
    def test_every_recorded_recipe_validates(self):
        paths = sorted((Path(__file__).parents[1] / "recipes").glob("*.json"))
        self.assertTrue(paths)
        run_ids = set()
        for path in paths:
            with self.subTest(recipe=path.name):
                spec = load_recipe(path)
                self.assertEqual(spec["name"], path.stem)
                self.assertNotIn(spec["run_id"], run_ids)
                run_ids.add(spec["run_id"])
                self.assertTrue(spec["provenance"]["results"])


if __name__ == "__main__":
    unittest.main()
