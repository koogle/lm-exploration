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

    def test_compiled_matches_eager(self):
        torch.manual_seed(2)
        z = torch.randn(32, 1000, device="cuda")
        targets = torch.randint(0, 1000, (32,), device="cuda")
        targets[0] = -100
        compiled = torch.compile(lambda z: self.kernel(z, targets, reduction="sum"), fullgraph=True)
        expected_loss, expected = grad(lambda z: self.kernel(z, targets, reduction="sum"), z)
        loss, actual = grad(compiled, z)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(actual, expected)

    def test_model_dense_and_chunked_losses_match_standard(self):
        from model_100m.model import GPT, ModelConfig

        from trainer.loss import cross_entropy

        torch.manual_seed(3)
        model = GPT(
            ModelConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64, eos_id=2)
        )
        model = model.cuda()
        x = torch.randint(0, 512, (4, 64), device="cuda")
        y = torch.randint(0, 512, (4, 64), device="cuda")
        y[1, 20:] = -100

        def run(**kwargs):
            model.zero_grad()
            loss = model(x, labels=y, **kwargs)
            loss.backward()
            return loss.detach(), [p.grad.clone() for p in model.parameters()]

        expected_loss, expected = run()
        for chunk in (0, 37):
            loss, actual = run(loss_chunk_size=chunk, loss_fn=cross_entropy)
            torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-4)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
