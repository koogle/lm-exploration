"""Agent mixes (`modal_scripts/modal_agent_mix.py`) and the reference family's schedule."""

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


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_to_the_floor(self):
        from trainer.reference import ReferenceFamily, RunConfig

        family = ReferenceFamily.__new__(ReferenceFamily)
        family.config = RunConfig(
            base="b", revision="r", recipe="x", target_tokens=1100, warmup_tokens=100,
            min_learning_rate_fraction=0.1,
        )  # fmt: skip
        family.target_tokens = 1100
        self.assertAlmostEqual(family.learning_rate_factor(50), 0.5)
        self.assertAlmostEqual(family.learning_rate_factor(100), 1.0)
        self.assertAlmostEqual(family.learning_rate_factor(600), 0.55)
        self.assertAlmostEqual(family.learning_rate_factor(1100), 0.1)


if __name__ == "__main__":
    unittest.main()
