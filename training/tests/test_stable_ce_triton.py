"""Run on CUDA, or on CPU with TRITON_INTERPRET=1 set before Python starts."""

import os
import unittest

import torch
from torch.nn import functional as F

from trainer import stable_softmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNNABLE = DEVICE == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"


def grad(fn, z):
    z = z.clone().requires_grad_()
    value = fn(z)
    (g,) = torch.autograd.grad(value, z)
    return value.detach(), g


@unittest.skipUnless(RUNNABLE, "needs CUDA or TRITON_INTERPRET=1")
class TritonCrossEntropyTests(unittest.TestCase):
    def setUp(self):
        from trainer.stable_ce_triton import cross_entropy

        self.kernel = cross_entropy

    def test_paper_example(self):
        z, t = torch.tensor([[9.0, -9.0, -9.0]], device=DEVICE), torch.tensor([0], device=DEVICE)
        loss, g = grad(lambda z: self.kernel(z, t), z)
        torch.testing.assert_close(loss.cpu(), torch.tensor(3.046e-8), rtol=1e-3, atol=0)
        expected = torch.tensor([[-3.046e-8, 1.523e-8, 1.523e-8]])
        torch.testing.assert_close(g.cpu(), expected, rtol=1e-3, atol=0)

    def test_matches_reference_across_blocks_and_margins(self):
        torch.manual_seed(0)
        # 5000 columns spans two 4096-wide blocks with a ragged tail.
        rows, vocab = 24, 5000
        z = torch.randn(rows, vocab, device=DEVICE)
        targets = torch.randint(0, vocab, (rows,), device=DEVICE)
        targets[[3, 11]] = -100
        targets[5] = vocab - 1
        z[torch.arange(rows), targets.clamp_min(0)] += torch.linspace(0, 80, rows, device=DEVICE)
        for reduction in ("none", "sum", "mean"):
            expected = stable_softmax.cross_entropy(z, targets, reduction=reduction)
            actual = self.kernel(z, targets, reduction=reduction)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)
        _, expected = grad(lambda z: stable_softmax.cross_entropy(z, targets, reduction="sum"), z)
        _, actual = grad(lambda z: self.kernel(z, targets, reduction="sum"), z)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-10)
        self.assertTrue(torch.equal(actual[3], torch.zeros_like(actual[3])))
        torch.testing.assert_close(
            actual.sum(-1), torch.zeros(rows, device=DEVICE), rtol=0, atol=1e-6
        )

    def test_bf16_matches_standard_on_ordinary_logits(self):
        torch.manual_seed(1)
        z = torch.randn(16, 300, device=DEVICE).bfloat16()
        targets = torch.randint(0, 300, (16,), device=DEVICE)
        expected_loss, expected = grad(lambda z: F.cross_entropy(z.float(), targets), z)
        loss, actual = grad(lambda z: self.kernel(z, targets), z)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
