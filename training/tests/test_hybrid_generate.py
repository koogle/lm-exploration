"""Incremental decoding matches the full forward pass."""

import unittest

import torch
from model_hybrid import generate, model

from tests.test_hybrid_backbone import tiny_config

EOS = tiny_config().eos_id


class GenerateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = tiny_config(n_layer=4, layer_pattern=("kda", "kda", "kda", "full"), mtp=False)
        self.lm = model.HybridLM(config).double().eval()

    def test_step_logits_match_full_forward(self):
        tokens = torch.randint(3, 37, (1, 12))
        full = self.lm(tokens)[0]
        cache = generate.Cache(self.lm)
        steps = torch.stack([generate.step(self.lm, int(t), cache) for t in tokens[0]])
        torch.testing.assert_close(steps.double(), full.double(), rtol=1e-6, atol=1e-6)

    def test_greedy_generation_continues_the_argmax_and_stops_at_eos(self):
        prompt = [5, 9, 11]
        out = list(generate.generate(self.lm, prompt, 6, EOS))
        tokens = list(prompt)
        for token in out:
            expected = int(self.lm(torch.tensor([tokens]))[0, -1].argmax())
            self.assertEqual(token, expected)
            tokens.append(token)
        self.assertTrue(len(out) == 6 or out[-1] == EOS)


if __name__ == "__main__":
    unittest.main()
