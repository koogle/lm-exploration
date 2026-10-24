"""Needs a CUDA GPU and nvcc; skipped otherwise."""

import unittest

import torch
from torch.nn import functional as F

from trainer import stable_softmax


def grad(fn, z):
    z = z.clone().requires_grad_()
    value = fn(z)
    (g,) = torch.autograd.grad(value, z)
    return value.detach(), g


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class CudaCrossEntropyTests(unittest.TestCase):
    def setUp(self):
        from trainer.stable_ce_cuda import cross_entropy

        self.kernel = cross_entropy

    def test_paper_example(self):
        z, t = torch.tensor([[9.0, -9.0, -9.0]], device="cuda"), torch.tensor([0], device="cuda")
        loss, g = grad(lambda z: self.kernel(z, t), z)
        torch.testing.assert_close(loss.cpu(), torch.tensor(3.046e-8), rtol=1e-3, atol=0)
        expected = torch.tensor([[-3.046e-8, 1.523e-8, 1.523e-8]])
        torch.testing.assert_close(g.cpu(), expected, rtol=1e-3, atol=0)

    def test_matches_reference_across_margins(self):
        torch.manual_seed(0)
        rows, vocab = 24, 5000
        z = torch.randn(rows, vocab, device="cuda", dtype=torch.float64)
        targets = torch.randint(0, vocab, (rows,), device="cuda")
        targets[[3, 11]] = -100
        targets[5] = vocab - 1
        z[torch.arange(rows), targets.clamp_min(0)] += torch.linspace(0, 80, rows, device="cuda")
        # The fp64 PyTorch port is the reference for the fp32 kernel.
        for reduction in ("none", "sum", "mean"):
            expected = stable_softmax.cross_entropy(z, targets, reduction=reduction)
            actual = self.kernel(z.float(), targets, reduction=reduction)
            torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=0)
        _, expected = grad(lambda z: stable_softmax.cross_entropy(z, targets, reduction="sum"), z)
        _, actual = grad(lambda z: self.kernel(z, targets, reduction="sum"), z.float())
        torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=0)
        self.assertTrue(torch.equal(actual[3], torch.zeros_like(actual[3])))
        torch.testing.assert_close(
            actual.sum(-1), torch.zeros(rows, device="cuda"), rtol=0, atol=1e-6
        )

    def test_bf16_and_fp64_match_standard_on_ordinary_logits(self):
        torch.manual_seed(1)
        targets = torch.randint(0, 32768, (64,), device="cuda")
        for dtype in (torch.bfloat16, torch.float64):
            z = torch.randn(64, 32768, device="cuda").to(dtype)
            expected_loss, expected = grad(lambda z: F.cross_entropy(z.float(), targets), z)
            loss, actual = grad(lambda z: self.kernel(z, targets), z)
            self.assertEqual(actual.dtype, dtype)
            torch.testing.assert_close(loss.float(), expected_loss)
            torch.testing.assert_close(actual.float(), expected.float())


if __name__ == "__main__":
    unittest.main()
