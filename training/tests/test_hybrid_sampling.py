"""Sampled decoding in `model_hybrid.generate`: temperature, top-k and top-p."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from model_hybrid import generate


class SampleTests(unittest.TestCase):
    def test_disabled_training_previews_skip_initial_periodic_and_final_sampling(self):
        from model_hybrid.train import HybridFamily

        family = HybridFamily.__new__(HybridFamily)
        family.config = SimpleNamespace(sample_every_steps=0)
        family.model_config = SimpleNamespace(eos_id=0, looped=False)
        model = torch.nn.Linear(1, 1)
        heldout = {"windows": {}, "tokenizer": None}
        with (
            patch("model_hybrid.train.evaluate", return_value={}),
            patch("model_hybrid.train.gate_sink_statistics", return_value={}),
            patch("model_hybrid.train.samples") as sampling,
        ):
            for step, final in [(0, False), (20, False), (21, True)]:
                result = family.evaluate(model, heldout, {"step": step}, final)
                self.assertNotIn("samples", result)
            sampling.assert_not_called()
            family.config.sample_every_steps = 20
            family.sampling = ([], (), 0)
            family.evaluate(model, heldout, {"step": 21}, True)
            sampling.assert_called_once()

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
