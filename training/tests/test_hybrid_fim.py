"""Fill-in-the-middle settings of hybrid run configs."""

import json
import tempfile
import unittest
from pathlib import Path

from model_hybrid import train

from tokenizer import glm

FIM = {"rate": 0.5, "spm_rate": 0.5, "sources": ["code"], "seed": 1}


def recipe():
    source = {"weight": 0.5, "max_passes": 1, "shards": 2}
    return {
        "name": "fim-test",
        "context": 16,
        "seed": 3,
        "sources": {"web": source, "code": source},
    }


class HybridFimTests(unittest.TestCase):
    def test_run_config_checks_fim(self):
        train.RunConfig(recipe="", model={}, target_tokens=1, fim=FIM)
        for bad in (
            {**FIM, "rate": 0},
            {**FIM, "spm_rate": 2},
            {**FIM, "sources": []},
            {"rate": 1},
        ):
            with self.assertRaises(ValueError):
                train.RunConfig(recipe="", model={}, target_tokens=1, fim=bad)

    def test_recipe_data_passes_fim_with_glm_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fim-test.json"
            path.write_text(json.dumps(recipe()))
            config = train.RunConfig(recipe=str(path), model={}, target_tokens=1, fim=FIM)
            identity, mixture, _, chat = train.recipe_data(config, folder)
            self.assertFalse(chat)
            self.assertEqual(identity["fim"], FIM)
            built = mixture()
            self.assertEqual(
                built.fim["tokens"],
                {
                    "prefix": glm.SPECIAL_TOKENS["<|code_prefix|>"],
                    "suffix": glm.SPECIAL_TOKENS["<|code_suffix|>"],
                    "middle": glm.SPECIAL_TOKENS["<|code_middle|>"],
                },
            )
            plain = train.recipe_data(
                train.RunConfig(recipe=str(path), model={}, target_tokens=1), folder
            )
            self.assertNotIn("fim", plain[0])
            self.assertIsNone(plain[1]().fim)


if __name__ == "__main__":
    unittest.main()
