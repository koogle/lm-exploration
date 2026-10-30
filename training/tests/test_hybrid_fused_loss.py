"""Fused output layer + cross-entropy + z-loss against the chunked autograd loss."""

import unittest

import torch
from model_hybrid import fused_loss, model


class FusedLossTests(unittest.TestCase):
    def test_values_and_gradients_match_autograd(self):
        torch.manual_seed(2)
        hidden = torch.randn(3, 7, 16, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(50, 16, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(0, 50, (3, 7))
        labels[1, 4:] = -100
        loss, ce, z = fused_loss.fused_cross_entropy(hidden, weight, labels, 1e-2, 5)
        expected_ce, expected_z = model.chunked_cross_entropy(hidden, weight, labels, 4)
        torch.testing.assert_close(ce, expected_ce)
        torch.testing.assert_close(z, expected_z)
        grads = torch.autograd.grad(2.5 * loss, (hidden, weight))
        expected = torch.autograd.grad(2.5 * (expected_ce + 1e-2 * expected_z), (hidden, weight))
        for actual, reference in zip(grads, expected, strict=True):
            torch.testing.assert_close(actual, reference)
        self.assertFalse(ce.requires_grad)


if __name__ == "__main__":
    unittest.main()
