"""Hybrid-family backbone: shapes, causality, GQA, initialization and loss."""

import math
import unittest

import torch
from model_hybrid import model
from torch.nn import functional as F


def tiny_config(**overrides):
    settings = {
        "vocab_size": 37,
        "eos_id": 2,
        "block_size": 16,
        "n_layer": 2,
        "n_embd": 32,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 8,
        "mlp_hidden": 48,
        "loss_chunk_size": 7,
    }
    return model.HybridConfig(**(settings | overrides))


def causal_mask(length):
    return model.DocumentMask(torch.zeros(1, length, dtype=torch.long))


class HybridBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_defaults(self):
        config = model.HybridConfig(154_880, eos_id=154_820)
        self.assertEqual((config.head_dim, config.rope_base, config.z_loss), (128, 1e6, 1e-4))

    def test_shapes_tied_embeddings_and_no_biases(self):
        lm = model.HybridLM(tiny_config())
        tokens = torch.randint(0, 37, (3, 11))
        self.assertEqual(lm(tokens).shape, (3, 11, 37))
        self.assertIs(lm.output_weight, lm.token_embedding.weight)
        biases = [name for name, _ in lm.named_parameters() if "bias" in name]
        self.assertTrue(all(name.endswith("mixer.gate_bias") for name in biases))
        self.assertTrue(all(m.bias is None for m in lm.modules() if isinstance(m, torch.nn.Linear)))
        with self.assertRaises(ValueError):
            lm(torch.zeros(1, 17, dtype=torch.long))

    def test_initialization(self):
        config = tiny_config(n_embd=256, mlp_hidden=512, n_head=8, n_kv_head=8, head_dim=32)
        lm = model.HybridLM(config)
        residual = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in lm.named_parameters():
            if parameter.ndim == 1:
                start = {"sink": 0.0, "gate_bias": 3.0}.get(name.rsplit(".", 1)[-1], 1.0)
                torch.testing.assert_close(parameter, torch.full_like(parameter, start))
                continue
            expected = residual if name.endswith(("o_proj.weight", "down.weight")) else 0.02
            self.assertAlmostEqual(parameter.std().item(), expected, delta=0.1 * expected)

    def test_rmsnorm_statistics_in_fp32(self):
        norm = model.RMSNorm(64)
        x = torch.randn(4, 64) * 300
        expected = F.rms_norm(x, (64,), norm.weight, eps=1e-5).bfloat16()
        torch.testing.assert_close(norm(x.bfloat16()), expected, rtol=1e-2, atol=1e-2)

    def test_causality(self):
        lm = model.HybridLM(tiny_config()).double()
        tokens = torch.randint(0, 37, (2, 12))
        changed = tokens.clone()
        changed[:, 7:] = torch.randint(0, 37, (2, 5))
        torch.testing.assert_close(lm(tokens)[:, :7], lm(changed)[:, :7], rtol=0, atol=0)
        self.assertFalse(torch.allclose(lm(tokens)[:, 7:], lm(changed)[:, 7:]))

    def test_mha_attention_matches_explicit_softmax(self):
        config = tiny_config(n_kv_head=4)
        attention = model.Attention(config).double()
        x = torch.randn(2, 9, 32, dtype=torch.float64)
        rope = tuple(t.double() for t in model.rope_tables(9, 8, config.rope_base, None))
        query, key, value = attention.project(x, rope)
        scores = query @ key.transpose(-1, -2) / math.sqrt(8)
        scores = scores.masked_fill(torch.ones(9, 9, dtype=torch.bool).triu(1), -math.inf)
        attended = (scores.softmax(-1) @ value) * torch.sigmoid(scores.logsumexp(-1))[..., None]
        gate = torch.sigmoid(attention.gate_proj(x) + 3.0)
        expected = attention.o_proj(attended.transpose(1, 2).reshape(2, 9, -1) * gate)
        torch.testing.assert_close(attention(x, rope, causal_mask(9)), expected)

    def test_gqa_matches_mha_with_shared_kv_heads(self):
        gqa = model.Attention(tiny_config()).double()
        mha = model.Attention(tiny_config(n_kv_head=4)).double()
        state = gqa.state_dict()
        for name in ("k_proj.weight", "v_proj.weight"):
            state[name] = state[name].view(2, 8, 32).repeat_interleave(2, 0).reshape(32, 32)
        mha.load_state_dict(state)
        x = torch.randn(2, 9, 32, dtype=torch.float64)
        rope = tuple(t.double() for t in model.rope_tables(9, 8, 1e6, None))
        torch.testing.assert_close(gqa(x, rope, causal_mask(9)), mha(x, rope, causal_mask(9)))

    def test_rope_base_changes_rotation(self):
        default = model.rope_tables(5, 8, 1e6, None)
        classic = model.rope_tables(5, 8, 1e4, None)
        torch.testing.assert_close(default[1][:, 0], classic[1][:, 0])
        torch.testing.assert_close(default[1][1, -1], torch.tensor(1e6**-0.75))
        torch.testing.assert_close(classic[1][1, -1], torch.tensor(1e4**-0.75))

    def test_chunked_loss_matches_full_cross_entropy_and_z_loss(self):
        lm = model.HybridLM(tiny_config()).double()
        tokens = torch.randint(0, 37, (2, 10))
        labels = torch.randint(0, 37, (2, 10))
        labels[1, 6:] = model.IGNORE_INDEX
        losses = lm(tokens, labels)
        logits = lm(tokens).flatten(0, 1)
        expected_ce = F.cross_entropy(logits, labels.flatten(), reduction="sum")
        lse = logits.logsumexp(-1)[labels.flatten() != model.IGNORE_INDEX]
        torch.testing.assert_close(losses["ce"], expected_ce)
        torch.testing.assert_close(losses["z"], lse.square().sum())
        torch.testing.assert_close(losses["loss"], expected_ce + 1e-4 * lse.square().sum())
        self.assertEqual(losses["count"].item(), 16)
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in lm.parameters()))


if __name__ == "__main__":
    unittest.main()
