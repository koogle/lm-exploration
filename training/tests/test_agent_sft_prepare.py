"""The hybrid's agent-mix subsets written while streaming read back as chat SFT data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from modal_scripts import modal_agent_sft
except ImportError:  # modal is not installed
    modal_agent_sft = None


@unittest.skipIf(modal_agent_sft is None, "modal is not installed")
class ChatSubsetTests(unittest.TestCase):
    def test_round_trip_through_chat_mixture(self):
        from training_data import chat_sft

        examples = {  # rank: (ids, mask, held out)
            2: ([5, 6, 7, 1], [0, 1, 1, 1], False),
            0: ([8, 9, 1], [0, 1, 1], False),
            1: ([3, 4, 1], [0, 1, 1], True),
            3: ([10, 11, 1], [0, 1, 1], False),
        }
        with tempfile.TemporaryDirectory() as folder:
            subset = modal_agent_sft._ChatSubset(folder, "src")
            for rank, (ids, mask, heldout) in examples.items():
                subset.write(rank, np.array(ids, np.uint32), np.array(mask, np.uint8), heldout)
            meta = subset.close(count=2)
            self.assertEqual(meta["conversations"], 2)
            self.assertEqual(meta["heldout_conversations"], 1)
            self.assertEqual(meta["stored_conversations"], 4)
            self.assertFalse(list(Path(folder).glob("*.raw")))

            tokens, mask, docs = chat_sft._load(folder, "src")
            # The two lowest-ranked training conversations, ranks 0 and 2.
            trained = sorted(tokens[s : s + n].tolist() for s, n in docs)
            self.assertEqual(trained, [[5, 6, 7, 1], [8, 9, 1]])
            held_tokens, held_mask, held_docs = chat_sft._load(folder, "src.heldout")
            self.assertEqual(held_tokens.tolist(), [3, 4, 1])
            self.assertEqual(held_mask.tolist(), [0, 1, 1])
            self.assertEqual(held_docs.tolist(), [[0, 3]])

            recipe = {"subsets": [{"name": "src", "repeat": 2}], "context": 16}
            mixture = chat_sft.ChatMixture(recipe, folder)
            self.assertEqual(len(mixture.order), 4)
            json.dumps(meta)


@unittest.skipIf(modal_agent_sft is None, "modal is not installed")
class GlmRecipeTests(unittest.TestCase):
    def test_recipes_match_their_mixes(self):
        for mix in ("smoke-200", "scale-500k"):
            recipe = modal_agent_sft.glm_recipe(mix)
            self.assertEqual(recipe["name"], f"agent-{mix}-glm")
            self.assertEqual(recipe["format"], "chat-sft")


if __name__ == "__main__":
    unittest.main()
