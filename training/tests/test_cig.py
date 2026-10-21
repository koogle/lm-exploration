import unittest

import torch
from torch.nn import functional as F

from trainer.cig import information_ratio, weighted_loss, weights


class CIGTests(unittest.TestCase):
    def test_gradient_matches_projected_soft_target(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 1, -100], [2, 3, 4]])
        c = torch.tensor([[0.0, 0.3, 1.0], [0.7, 1.0, 0.01]], dtype=torch.float64)
        p = logits.softmax(-1).detach()
        tilted = p.clone()
        for b, t in (targets != -100).nonzero().tolist():
            tilted[b, t, targets[b, t]] *= torch.exp(c[b, t] / 0.3)
        q = tilted / tilted.sum(-1, keepdim=True)
        soft = -(q * logits.log_softmax(-1)).sum(-1)[targets != -100].mean()
        (expected,) = torch.autograd.grad(soft, logits, retain_graph=True)
        value, count = weighted_loss(logits, targets, c)
        (actual,) = torch.autograd.grad(value / count, logits)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(count, 5)
        self.assertTrue(torch.equal(actual[0, 2], torch.zeros(5)))
        self.assertTrue(torch.equal(actual[0, 0], torch.zeros(5)))

    def test_frozen_information_and_stable_weights(self):
        a = torch.tensor([2.0, 3.0, 0.0], requires_grad=True)
        b = torch.tensor([4.0, 2.0, 0.0], requires_grad=True)
        c = information_ratio(a, b)
        torch.testing.assert_close(c, torch.tensor([0.5, 0.0, 0.0]))
        self.assertFalse(c.requires_grad)
        w = weights(torch.tensor([1e4, 0.0, 1.0]), torch.tensor([1.0, 1.0, 0.0]), 0.01)
        self.assertTrue(torch.isfinite(w).all())
        torch.testing.assert_close(w, torch.tensor([0.0, 1.0, 0.0]))

    def test_unweighted_token_denominator(self):
        logits = torch.zeros(1, 2, 3, requires_grad=True)
        y = torch.tensor([[1, 2]])
        c = torch.tensor([[1.0, 0.0]])
        value, count = weighted_loss(logits, y, c)
        expected = (
            F.cross_entropy(logits[0, :1], y[0, :1])
            * weights(torch.tensor(3.0).log(), torch.tensor(1.0))
            / 2
        )
        torch.testing.assert_close(value / count, expected)


if __name__ == "__main__":
    unittest.main()
