import importlib.util
import json
import sys
import unittest
from pathlib import Path

import model_10m.model as backbone
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_generation():
    # The Space ships the decoder as `model.py`; here the installed copy stands in.
    sys.modules["model"] = backbone
    spec = importlib.util.spec_from_file_location(
        "playground_generation", ROOT / "analysis/playground/generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlaygroundGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = backbone.ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)
        self.model = backbone.GPT(config).eval()
        self.generation = load_generation()

    def tearDown(self):
        sys.modules.pop("model", None)

    def test_cached_steps_match_the_full_forward(self):
        tokens = [1, 5, 9, 12, 3, 7, 20, 4]
        with torch.no_grad():
            full = self.model(torch.tensor([tokens]))[0]
        cache = [None] * len(self.model.blocks)
        logits = self.generation.step(self.model, tokens[:5], cache, 0)
        torch.testing.assert_close(logits, full[4], atol=1e-5, rtol=1e-5)
        for position in range(5, len(tokens)):
            logits = self.generation.step(self.model, [tokens[position]], cache, position)
            torch.testing.assert_close(logits, full[position], atol=1e-5, rtol=1e-5)

    def test_greedy_completion_matches_uncached_greedy(self):
        prompt = [1, 5, 9]
        generated = self.generation.complete(self.model, prompt, 6, 0, 0, 1.0, 1.0, 0, -1, 32)
        expected, sequence = [], list(prompt)
        with torch.no_grad():
            for _ in range(6):
                token = int(self.model(torch.tensor([sequence]))[0, -1].argmax())
                expected.append(token)
                sequence.append(token)
        self.assertEqual(generated, expected)

    def test_nucleus_keeps_the_crossing_token(self):
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        draws = {
            self.generation.choose(logits, [], 1.0, 0, 0.6, 1.0, torch.Generator().manual_seed(s))
            for s in range(200)
        }
        self.assertEqual(draws, {0, 1})

    def test_eos_and_stop_end_generation(self):
        first = self.generation.complete(self.model, [1, 5], 4, 0, 0, 1.0, 1.0, 0, -1, 32)[0]
        self.assertEqual(
            self.generation.complete(self.model, [1, 5], 4, 0, 0, 1.0, 1.0, 0, first, 32), []
        )
        stopped = self.generation.complete(
            self.model, [1, 5], 4, 0, 0, 1.0, 1.0, 0, -1, 32, stop=lambda g: len(g) == 2
        )
        self.assertEqual(len(stopped), 2)


class ConfigTests(unittest.TestCase):
    def test_layout_shows_every_model(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "analysis/playground/configs/base-vs-math-sft.json").read_text()
        )
        labels = sorted(m["label"] for m in config["models"])
        self.assertEqual(labels, sorted(label for row in config["layout"] for label in row))


if __name__ == "__main__":
    unittest.main()
