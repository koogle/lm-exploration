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
        steps = torch.stack([generate.step(self.lm, [int(t)], cache)[0] for t in tokens[0]])
        torch.testing.assert_close(steps.double(), full.double(), rtol=1e-6, atol=1e-6)

    def test_rows_cross_documents_like_the_full_forward(self):
        tokens = torch.randint(3, 37, (3, 14))
        tokens[0, 4] = EOS
        tokens[1, 0] = tokens[1, 9] = EOS
        full = self.lm(tokens)
        cache = generate.Cache(self.lm, rows=3)
        steps = torch.stack([generate.step(self.lm, column, cache) for column in tokens.T], 1)
        torch.testing.assert_close(steps.double(), full.double(), rtol=1e-6, atol=1e-6)

    def test_prefill_continues_like_the_full_forward(self):
        prompts = [[5, 9, 11, 4, 20, 6], [7, 3], [12, 3, 30, 8]]
        following = [[17, 2], [21, 22], [8, 8]]
        logits, cache = generate.prefill(self.lm, prompts)
        steps = [generate.step(self.lm, [f[i] for f in following], cache) for i in range(2)]
        for row, (prompt, after) in enumerate(zip(prompts, following, strict=True)):
            full = self.lm(torch.tensor([prompt + after]))[0]
            last = len(prompt) - 1
            torch.testing.assert_close(
                logits[row].double(), full[last].double(), rtol=1e-5, atol=1e-5
            )
            for i, logit in enumerate(steps):
                torch.testing.assert_close(
                    logit[row].double(), full[last + 1 + i].double(), rtol=1e-5, atol=1e-5
                )

    def test_batched_generation_matches_one_row_at_a_time(self):
        prompts = [[5, 9, 11], [7], [12, 3, 30, 8, 4]]
        batched = generate.generate_batch(self.lm, prompts, 5, (EOS,))
        alone = [list(generate.generate(self.lm, p, 5, (EOS,))) for p in prompts]
        self.assertEqual(batched, alone)

    def test_greedy_generation_continues_the_argmax_and_stops_at_eos(self):
        prompt = [5, 9, 11]
        out = list(generate.generate(self.lm, prompt, 6, (EOS,)))
        tokens = list(prompt)
        for token in out:
            expected = int(self.lm(torch.tensor([tokens]))[0, -1].argmax())
            self.assertEqual(token, expected)
            tokens.append(token)
        self.assertTrue(len(out) == 6 or out[-1] == EOS)


if __name__ == "__main__":
    unittest.main()
