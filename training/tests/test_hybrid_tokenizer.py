"""The pinned GLM tokenizer and the recipe that uses it."""

import json
import os
import unittest
from pathlib import Path

from model_hybrid import data, model, tokenizer

RECIPE = Path(__file__).parents[1] / "hybrid" / "recipes" / "hybrid-40b-v1.json"
# Set HYBRID_GLM_TOKENIZER to a downloaded tokenizer.json to check the file itself.
LOCAL = os.environ.get("HYBRID_GLM_TOKENIZER", "artifacts/glm/GLM-4.7-Flash/tokenizer.json")


class TokenizerRecipeTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(tokenizer.SPECIAL_TOKENS["<|endoftext|>"], tokenizer.EOS_ID)
        self.assertGreaterEqual(tokenizer.VOCAB_SIZE, tokenizer.TOKENIZER_SIZE)
        self.assertEqual(tokenizer.VOCAB_SIZE % 64, 0)
        self.assertLess(max(tokenizer.SPECIAL_TOKENS.values()), tokenizer.TOKENIZER_SIZE)

    def test_recipe(self):
        recipe = data.load_recipe(RECIPE)
        self.assertEqual(recipe["tokenizer_sha256"], tokenizer.SHA256)
        self.assertEqual(recipe["target_tokens"], 40_000_000_000)
        weights = {name: s["weight"] for name, s in recipe["sources"].items()}
        self.assertEqual(weights, {"climbmix": 0.7, "code": 0.2, "math": 0.1})
        for name, source in recipe["sources"].items():
            order = data.training_order(recipe, name)
            self.assertNotIn(data.heldout_index(source), order)
            self.assertEqual(sorted(order), list(range(source["shards"] - 1)))
            # Planned unique tokens cover the source's share in one pass.
            planned = len(order) * source["estimated_tokens_per_shard"]
            self.assertGreater(planned, recipe["target_tokens"] * source["weight"] * 1.05)
        self.assertEqual(model.HybridConfig(tokenizer.VOCAB_SIZE).vocab_size, 154_880)

    @unittest.skipUnless(Path(LOCAL).exists(), "GLM tokenizer.json not downloaded")
    def test_tokenizer_file(self):
        glm = tokenizer.load(LOCAL)
        ids = glm.encode("def f(x):\n    return x", add_special_tokens=False).ids
        self.assertNotIn(tokenizer.EOS_ID, ids)
        config = json.loads(Path(LOCAL).read_text())
        self.assertEqual(len(config["added_tokens"]), 36)


if __name__ == "__main__":
    unittest.main()
