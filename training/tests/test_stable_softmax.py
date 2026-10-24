import unittest

import torch
from torch.nn import functional as F

from trainer.stable_softmax import cross_entropy, log_softmax, soft_target_cross_entropy, softmax


def grad(fn, z):
    z = z.clone().requires_grad_()
    value = fn(z)
    (g,) = torch.autograd.grad(value, z)
    return value.detach(), g


class SoftmaxCollapseTests(unittest.TestCase):
    def test_paper_example(self):
        z, t = torch.tensor([[9.0, -9.0, -9.0]]), torch.tensor([0])
        # The bug: standard fp32 rounds the loss and target gradient to zero.
        loss, g = grad(lambda z: F.cross_entropy(z, t), z)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(g[0, 0]), 0.0)
        self.assertGreater(float(g.sum()), 0.0)

        loss, g = grad(lambda z: cross_entropy(z, t), z)
        torch.testing.assert_close(loss, torch.tensor(3.046e-8), rtol=1e-3, atol=0)
        torch.testing.assert_close(
            g, torch.tensor([[-3.046e-8, 1.523e-8, 1.523e-8]]), rtol=1e-3, atol=0
        )
        self.assertEqual(float(g.sum()), 0.0)
        torch.testing.assert_close(log_softmax(z)[0, 0], torch.tensor(-3.046e-8), rtol=1e-3, atol=0)
        _, g = grad(lambda z: softmax(z)[0, 0], z)
        torch.testing.assert_close(
            g, torch.tensor([[3.046e-8, -1.523e-8, -1.523e-8]]), rtol=1e-3, atol=0
        )

    def test_fp32_matches_fp64_past_the_cliff(self):
        torch.manual_seed(0)
        base = torch.randn(64, 50, dtype=torch.float64)
        targets = torch.randint(0, 50, (64,))
        # Margins from about 0 to 80 nats, beyond fp32's 16.6 cliff.
        base[torch.arange(64), targets] += torch.linspace(0, 80, 64, dtype=torch.float64)
        ref_loss, ref_g = grad(lambda z: F.cross_entropy(z, targets, reduction="sum"), base)
        loss, g = grad(lambda z: cross_entropy(z, targets, reduction="sum"), base.float())
        torch.testing.assert_close(loss.double(), ref_loss, rtol=1e-5, atol=0)
        # Standard fp64 collapses too past m = 36.7, so use the closed form (eq. 7).
        rows = torch.arange(64)
        others = base.clone()
        others[rows, targets] = float("-inf")
        m = base[rows, targets] - others.logsumexp(-1)
        exact = -torch.exp(-m - F.softplus(-m))
        torch.testing.assert_close(g[rows, targets].double(), exact, rtol=1e-4, atol=0)
        self.assertTrue(torch.equal(ref_g[rows, targets][-1], torch.zeros((), dtype=torch.float64)))
        torch.testing.assert_close(g.sum(-1), torch.zeros(64), rtol=0, atol=1e-7)

    def test_matches_standard_on_ordinary_logits(self):
        torch.manual_seed(1)
        z = torch.randn(10, 7, dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, -100, 4, 5, 6, 0, -100, 3])
        for reduction in ("mean", "sum", "none"):
            expected = F.cross_entropy(z, targets, reduction=reduction)
            torch.testing.assert_close(cross_entropy(z, targets, reduction=reduction), expected)
        _, expected = grad(lambda z: F.cross_entropy(z, targets), z)
        _, actual = grad(lambda z: cross_entropy(z, targets), z)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual[3], torch.zeros(7)))
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        torch.testing.assert_close(log_softmax(x, dim=1), F.log_softmax(x, dim=1))
        torch.testing.assert_close(softmax(x, dim=1), F.softmax(x, dim=1))

    def test_bf16_logits_are_computed_in_fp32(self):
        z = torch.tensor([[9.0, -9.0, -9.0]]).bfloat16()
        loss = cross_entropy(z, torch.tensor([0]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(float(loss), 0.0)


class SoftmaxCancellationTests(unittest.TestCase):
    def test_paper_example(self):
        z_t = torch.tensor([2.0, 1.0, 0.0])
        z_s = z_t + torch.tensor([-1e-7, 1e-7, 0.0])
        exact = torch.softmax(z_s.double(), -1) - torch.softmax(z_t.double(), -1)
        _, g = grad(lambda z: soft_target_cross_entropy(z, z_t), z_s)
        torch.testing.assert_close(g.double(), exact, rtol=1e-3, atol=0)
        self.assertGreater(float(g[2]), 0.0)

    def test_zero_gradient_at_teacher(self):
        torch.manual_seed(2)
        z = torch.randn(4, 6, 32) * 5
        _, g = grad(lambda s: soft_target_cross_entropy(s, z).sum(), z)
        self.assertTrue(torch.equal(g, torch.zeros_like(g)))

    def test_matches_standard_away_from_teacher(self):
        torch.manual_seed(3)
        z_s, z_t = torch.randn(5, 9, dtype=torch.float64), torch.randn(5, 9, dtype=torch.float64)
        standard = lambda z: -(F.softmax(z_t, -1) * F.log_softmax(z, -1)).sum(-1).sum()
        expected_loss, expected = grad(standard, z_s)
        loss, actual = grad(lambda z: soft_target_cross_entropy(z, z_t).sum(), z_s)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(actual, expected)
        _, actual = grad(lambda z: soft_target_cross_entropy(z.T, z_t.T, dim=0).sum(), z_s)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
