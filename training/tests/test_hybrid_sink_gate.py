"""Attention sink and output gate in the hybrid family."""

import math
import unittest

import torch
from model_hybrid import model

from tests.test_hybrid_backbone import causal_mask, tiny_config


class SinkGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.attention = model.Attention(tiny_config()).double()
        self.x = torch.randn(2, 9, 32, dtype=torch.float64)
        self.rope = tuple(t.double() for t in model.rope_tables(9, 8, 1e6, None))

    def test_sink_matches_softmax_with_an_extra_zero_value_key(self):
        sink = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=torch.float64)
        query, key, value = self.attention.project(self.x, self.rope)
        key, value = key.repeat_interleave(2, 1), value.repeat_interleave(2, 1)
        scores = query @ key.transpose(-1, -2) / math.sqrt(8)
        scores = scores.masked_fill(torch.ones(9, 9, dtype=torch.bool).triu(1), -math.inf)
        sink_column = sink[None, :, None, None].expand(2, 4, 9, 1)
        weights = torch.cat((scores, sink_column), -1).softmax(-1)
        zero = torch.zeros(2, 4, 1, 8, dtype=torch.float64)
        expected = weights[..., :9] @ value + weights[..., 9:] @ zero

        attended, lse = model.reference_attention(query, key, value, causal_mask(9).dense())
        actual = attended * torch.sigmoid(lse - sink[:, None])[..., None]
        torch.testing.assert_close(actual, expected)

        # The same through the module, with the gate and output projection.
        with torch.no_grad():
            self.attention.sink.copy_(sink)
        gate = torch.sigmoid(self.attention.gate_proj(self.x) + self.attention.gate_bias)
        merged = expected.transpose(1, 2).reshape(2, 9, -1) * gate
        torch.testing.assert_close(
            self.attention(self.x, self.rope, causal_mask(9)), self.attention.o_proj(merged)
        )

    def test_open_gate_and_absent_sink_reproduce_plain_attention(self):
        with torch.no_grad():
            self.attention.sink.fill_(-math.inf)
            self.attention.gate_bias.fill_(math.inf)
        query, key, value = self.attention.project(self.x, self.rope)
        attended, _ = model.reference_attention(query, key, value, causal_mask(9).dense())
        plain = self.attention.o_proj(attended.transpose(1, 2).reshape(2, 9, -1))
        torch.testing.assert_close(
            self.attention(self.x, self.rope, causal_mask(9)), plain, rtol=0, atol=0
        )

    def test_gate_starts_mostly_open_and_learns(self):
        lm = model.HybridLM(tiny_config())
        attention = lm.blocks[0].mixer
        gate = torch.sigmoid(attention.gate_proj(torch.randn(4, 32)) + attention.gate_bias)
        self.assertGreater(gate.mean().item(), 0.9)
        tokens = torch.randint(0, 37, (2, 8))
        lm(tokens, torch.randint(0, 37, (2, 8)))["loss"].backward()
        for parameter in (attention.sink, attention.gate_bias, attention.gate_proj.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
