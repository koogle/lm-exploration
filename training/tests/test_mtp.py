"""Alignment, leakage and gradient checks for the experimental MTP model."""

import unittest

import torch
from model_10m.model import ModelConfig, find_documents
from model_10m.mtp import GPTWithMTP
from torch.nn import functional as F


class MTPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = GPTWithMTP(ModelConfig(19, 8, 1, 2, 16, eos_id=18))
        self.sequence = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
        self.lengths = torch.tensor([5, 3])

    def test_targets_counts_and_shared_projection(self):
        model, seq = self.model, self.sequence
        losses = model.training_losses(seq, self.lengths)
        self.assertEqual(losses["ntp_count"].item(), 6)
        self.assertEqual(losses["mtp_count"].item(), 4)
        hidden, _ = model.trunk(seq[:, :-1])
        documents = find_documents(seq[:, :-2], model.config.eos_id)
        future = model.mtp(hidden[:, :-1], model.token_embedding(seq[:, 1:-1]), documents)
        logits = model.lm_head(future)
        expected = F.cross_entropy(logits[0], seq[0, 2:], reduction="sum")
        expected += F.cross_entropy(logits[1, :1], seq[1, 2:3], reduction="sum")
        torch.testing.assert_close(losses["mtp_sum"], expected)
        ntp = model(seq[:, :-1])
        expected_ntp = F.cross_entropy(ntp[0], seq[0, 1:], reduction="sum")
        expected_ntp += F.cross_entropy(ntp[1, :2], seq[1, 1:3], reduction="sum")
        torch.testing.assert_close(losses["ntp_sum"], expected_ntp)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        loss = losses["ntp_sum"] / 6 + 0.1 * losses["mtp_sum"] / 4
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.mtp.fuse.weight.grad.abs().sum().item(), 0)

    def test_future_target_cannot_leak_into_prediction(self):
        def future_logits(seq):
            tokens = seq[:, :-1]
            hidden, _ = self.model.trunk(tokens)
            documents = find_documents(tokens[:, :-1], self.model.config.eos_id)
            future = self.model.mtp(
                hidden[:, :-1], self.model.token_embedding(tokens[:, 1:]), documents
            )
            return self.model.lm_head(future)

        changed = self.sequence.clone()
        changed[:, 3:] = 12
        # Positions 0 and 1 predict x_2 and x_3; neither can see x_3.
        torch.testing.assert_close(
            future_logits(self.sequence)[:, :2], future_logits(changed)[:, :2], rtol=0, atol=0
        )

    def test_padding_and_empty_auxiliary_targets(self):
        changed = self.sequence.clone()
        changed[1, 3:] = 13
        original = self.model.training_losses(self.sequence, self.lengths)
        padded = self.model.training_losses(changed, self.lengths)
        for key in original:
            torch.testing.assert_close(original[key], padded[key])
        empty = self.model.training_losses(self.sequence, torch.tensor([2, 1]))
        self.assertEqual(empty["mtp_count"].item(), 0)
        self.assertEqual(empty["mtp_sum"].item(), 0)
        empty["mtp_sum"].backward()
        self.assertIsNotNone(self.model.mtp.fuse.weight.grad)


if __name__ == "__main__":
    unittest.main()
