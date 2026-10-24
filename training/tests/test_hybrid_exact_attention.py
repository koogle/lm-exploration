"""The fixed-order attention definition the CUDA kernel must match bit for bit."""

import math
import unittest

import torch
from model_hybrid import documents, exact_attention, model

EOS = 2


def packed_starts(length, ends):
    tokens = torch.full((1, length), 5)
    tokens[0, ends] = EOS
    return exact_attention.document_starts(documents.document_ids(tokens, EOS))


class ExactAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_exp_and_log(self):
        x = torch.linspace(-86, 0, 200_001, dtype=torch.float32)
        expected = torch.exp(x.double())
        relative = ((exact_attention.exp_fixed(x).double() - expected) / expected).abs()
        # Rounding y = x * log2(e) costs half an ulp of y, so the bound grows with |x|.
        self.assertTrue((relative < 1.2e-7 * x.double().abs().clamp(min=1)).all())
        self.assertEqual(exact_attention.exp_fixed(torch.tensor([0.0])).item(), 1.0)
        self.assertEqual(exact_attention.exp_fixed(torch.tensor([-90.0, -math.inf])).sum(), 0)
        y = torch.logspace(0, 5, 200_001, dtype=torch.float32)
        error = (exact_attention.log_fixed(y).double() - torch.log(y.double())).abs()
        self.assertLess(error.max().item(), 2e-6)
        self.assertEqual(exact_attention.log_fixed(torch.tensor([1.0])).item(), 0.0)

    def test_document_starts(self):
        starts = packed_starts(9, [2, 5])
        self.assertEqual(starts.tolist(), [[0, 0, 0, 3, 3, 3, 6, 6, 6]])
        self.assertEqual(starts.dtype, torch.int32)

    def inputs(self, length=70, heads=4, kv_heads=2, width=16):
        q = torch.randn(1, heads, length, width)
        k = torch.randn(1, kv_heads, length, width)
        v = torch.randn(1, kv_heads, length, width)
        return q, k, v, packed_starts(length, [9, 40])

    def test_forward_matches_softmax(self):
        q, k, v, starts = self.inputs()
        scale = 1 / math.sqrt(16)
        out, lse, m, total = exact_attention.exact_attention(q, k, v, starts, scale)
        allowed = exact_attention._allowed(starts, 70)
        expected, expected_lse = model.reference_attention(
            q.double(), k.double(), v.double(), allowed
        )
        torch.testing.assert_close(out.double(), expected, rtol=0, atol=2e-6)
        torch.testing.assert_close(lse.double(), expected_lse, rtol=0, atol=2e-6)
        torch.testing.assert_close(lse, m + torch.log(total), rtol=0, atol=1e-6)

    def test_backward_matches_autograd(self):
        q, k, v, starts = self.inputs()
        scale = 1 / math.sqrt(16)
        out, lse, m, total = exact_attention.exact_attention(q, k, v, starts, scale)
        dout, dlse = torch.randn_like(out), torch.randn_like(lse)
        grads = exact_attention.exact_attention_backward(
            q, k, v, starts, scale, out, dout, dlse, m, total
        )
        leaves = [t.double().requires_grad_() for t in (q, k, v)]
        allowed = exact_attention._allowed(starts, 70)
        expected, expected_lse = model.reference_attention(*leaves, allowed)
        objective = (expected * dout.double()).sum() + (expected_lse * dlse.double()).sum()
        for actual, reference in zip(grads, torch.autograd.grad(objective, leaves), strict=True):
            torch.testing.assert_close(actual.double(), reference, rtol=0, atol=2e-5)

    def test_bfloat16_rounds_outputs_only_at_the_end(self):
        q, k, v, starts = (t.bfloat16() if t.is_floating_point() else t for t in self.inputs())
        out, lse, *_ = exact_attention.exact_attention(q, k, v, starts, 0.25)
        reference = exact_attention.exact_attention(q.float(), k.float(), v.float(), starts, 0.25)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, reference[0].bfloat16()))
        self.assertTrue(torch.equal(lse, reference[1]))


if __name__ == "__main__":
    unittest.main()
