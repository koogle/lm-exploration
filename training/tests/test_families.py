"""Family adapters: found by model module; prompts as a user turn or as plain text."""

import unittest
from types import SimpleNamespace

from trainer import families


class Characters:
    """A stand-in tokenizer: one id per character."""

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) for c in text])


class FamilyTests(unittest.TestCase):
    def test_adapters_by_model_module(self):
        self.assertEqual(families.adapter("model_100m.model").name, "subword")
        self.assertEqual(families.adapter("model_hybrid.model").name, "hybrid")
        with self.assertRaisesRegex(ValueError, "No post-training adapter"):
            families.adapter("model_10m.model")

    def test_hybrid_text_is_plain_and_prompts_are_chat_turns(self):
        family = families.adapter("model_hybrid.model")
        self.assertEqual(family.text_ids(Characters(), "def f"), [ord(c) for c in "def f"])
        turn = "".join(map(chr, family.prompt_ids(Characters(), "def f")))
        self.assertIn("def f", turn)
        self.assertNotEqual(turn, "def f")

    def test_subword_text_and_prompts_start_with_bos(self):
        family = families.adapter("model_100m.model")
        ids = family.text_ids(Characters(), "ab")
        self.assertEqual(ids, family.prompt_ids(Characters(), "ab"))
        self.assertEqual(ids[1:], [ord("a"), ord("b")])
        self.assertEqual(ids[0], family.special_tokens["bos"])


if __name__ == "__main__":
    unittest.main()
