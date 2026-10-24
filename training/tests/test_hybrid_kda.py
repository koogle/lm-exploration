"""Kimi Delta Attention layers and the hybrid layer layout."""

import unittest

import torch
from model_hybrid import documents, kda, model
from torch.nn import functional as F

from tests.test_hybrid_backbone import tiny_config

EOS = 2


def matrix_form(q, k, v, g, beta, A_log, dt_bias):
    """One sequence, written as S_t = (I - b k k^T) diag(a) S_(t-1) + b k v^T."""
    length, heads, width = k.shape
    outputs = torch.zeros(length, heads, v.size(-1), dtype=torch.float64)
    for h in range(heads):
        state = torch.zeros(width, v.size(-1), dtype=torch.float64)
        for t in range(length):
            key = k[t, h] / k[t, h].norm()
            query = q[t, h] / q[t, h].norm() / width**0.5
            rate = A_log[h].exp() * F.softplus(g[t, h] + dt_bias.view(heads, width)[h])
            strength = torch.sigmoid(beta[t, h])
            householder = torch.eye(width, dtype=torch.float64) - strength * torch.outer(key, key)
            state = householder @ torch.diag((-rate).exp()) @ state
            state = state + strength * torch.outer(key, v[t, h])
            outputs[t, h] = state.T @ query
    return outputs


def random_inputs(batch, length, heads=2, width=4):
    shape = (batch, length, heads, width)
    return (
        torch.randn(shape, dtype=torch.float64),
        torch.randn(shape, dtype=torch.float64),
        torch.randn(shape, dtype=torch.float64),
        torch.randn(shape, dtype=torch.float64),
        torch.randn(batch, length, heads, dtype=torch.float64),
        torch.rand(heads, dtype=torch.float64).log1p(),
        torch.randn(heads * width, dtype=torch.float64),
    )


class KDATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_default_layout(self):
        config = model.HybridConfig(154_880)
        kinds = config.layer_kinds
        self.assertEqual([i for i, kind in enumerate(kinds) if kind == "full"], [3, 7, 11, 15])
        self.assertEqual(kinds.count("kda"), 12)
        with self.assertRaises(ValueError):
            model.HybridConfig(10, n_layer=6)
        with self.assertRaises(ValueError):
            model.HybridConfig(10, layer_pattern=("kda", "mamba"))

    def test_reference_matches_matrix_form(self):
        q, k, v, g, beta, A_log, dt_bias = random_inputs(2, 7)
        positions = torch.arange(7).expand(2, 7)
        actual = kda.kda_reference(q, k, v, g, beta, A_log, dt_bias, positions)
        for row in range(2):
            expected = matrix_form(q[row], k[row], v[row], g[row], beta[row], A_log, dt_bias)
            torch.testing.assert_close(actual[row], expected)

    def test_reference_resets_state_at_documents(self):
        inputs = random_inputs(1, 9)
        q, k, v, g, beta, A_log, dt_bias = inputs
        ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]])
        positions = documents.document_positions(ids)
        packed = kda.kda_reference(q, k, v, g, beta, A_log, dt_bias, positions)
        alone = kda.kda_reference(
            q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:], A_log, dt_bias, positions[:, 4:]
        )
        torch.testing.assert_close(packed[:, 4:], alone, rtol=0, atol=0)

    def test_short_convolution_matches_conv1d_and_resets(self):
        x = torch.randn(2, 10, 6, dtype=torch.float64)
        weight = torch.randn(6, 1, 4, dtype=torch.float64)
        expected = F.silu(F.conv1d(F.pad(x.transpose(1, 2), (3, 0)), weight, groups=6))
        positions = torch.arange(10).expand(2, 10)
        actual = kda.short_convolution(x, weight, positions)
        torch.testing.assert_close(actual, expected.transpose(1, 2))
        ids = torch.tensor([[0] * 4 + [1] * 6] * 2)
        packed = kda.short_convolution(x, weight, documents.document_positions(ids))
        alone = kda.short_convolution(x[:, 4:], weight, positions[:, :6])
        torch.testing.assert_close(packed[:, 4:], alone, rtol=0, atol=0)

    def test_hybrid_model_isolates_documents(self):
        lm = model.HybridLM(tiny_config(n_layer=4, layer_pattern=("kda", "full"))).double()
        self.assertEqual([b.kind for b in lm.blocks], ["kda", "full", "kda", "full"])
        b = torch.tensor([11, 12, 13, 14, 15])
        first = torch.cat((torch.tensor([20, 21, 22, EOS]), b))[None]
        second = torch.cat((torch.tensor([30, 31, 32, 33, 34, 35, EOS]), b))[None]
        outputs = [lm(x, documents=documents.document_ids(x, EOS)) for x in (first, second)]
        torch.testing.assert_close(outputs[0][0, 4:], outputs[1][0, 7:], rtol=0, atol=1e-12)
        torch.testing.assert_close(outputs[0][0, 4:], lm(b[None])[0], rtol=0, atol=1e-12)

    def test_hybrid_model_is_causal_and_trains(self):
        lm = model.HybridLM(tiny_config(n_layer=4, layer_pattern=("kda", "kda", "kda", "full")))
        lm = lm.double()
        tokens = torch.randint(0, 37, (2, 12))
        changed = tokens.clone()
        changed[:, 7:] = torch.randint(0, 37, (2, 5))
        torch.testing.assert_close(lm(tokens)[:, :7], lm(changed)[:, :7], rtol=0, atol=0)
        self.assertFalse(torch.allclose(lm(tokens)[:, 7:], lm(changed)[:, 7:]))
        lm(tokens, torch.randint(0, 37, (2, 12)))["loss"].backward()
        for name, parameter in lm.named_parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
