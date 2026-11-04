"""Agent mixes (`modal_scripts/modal_agent_mix.py`)."""

import unittest

try:
    from modal_scripts import modal_agent_mix
except ImportError:  # modal is not installed
    modal_agent_mix = None


@unittest.skipIf(modal_agent_mix is None, "modal is not installed")
class RecipeTests(unittest.TestCase):
    def test_recipes_match_their_mixes(self):
        for mix in ("smoke-200", "scale-500k"):
            for chat in modal_agent_mix.CHAT_FORMATS:
                recipe = modal_agent_mix.mix_recipe(mix, chat)
                self.assertEqual(recipe["name"], f"agent-{mix}-{chat}")
        qwen = modal_agent_mix.mix_recipe("scale-500k", "qwen")
        self.assertEqual(qwen["eos_id"], 151643)


if __name__ == "__main__":
    unittest.main()
