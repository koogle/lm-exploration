"""Sampled decoding in `model_hybrid.generate`: temperature, top-k and top-p."""

import unittest

import torch
from model_hybrid import generate


class SampleTests(unittest.TestCase):
    def draws(self, logits, **options):
        generator = torch.Generator().manual_seed(0)
        seen = set()
        for _ in range(300):
            token = generate._sample(logits, 1.0, options.get("top_p", 1.0), generator,
                                     options.get("top_k", 0))  # fmt: skip
            seen.update(enumerate(token.tolist()))
        return seen

    def test_top_k_keeps_the_k_most_likely(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, -9.0]])
        self.assertEqual(self.draws(logits, top_k=2), {(0, 0), (0, 1)})

    def test_top_p_keeps_the_nucleus(self):
        logits = torch.tensor([[9.0, 0.0, 0.0, 0.0]])
        self.assertEqual(self.draws(logits, top_p=0.5), {(0, 0)})

    def test_greedy_at_temperature_zero(self):
        logits = torch.tensor([[0.0, 3.0, 1.0]])
        self.assertEqual(generate._sample(logits, 0.0, 1.0, None).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
