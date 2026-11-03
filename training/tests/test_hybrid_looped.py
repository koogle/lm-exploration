"""Looped (shared recurrent block) hybrid models: layout, reinjection and accounting."""

import unittest

import torch
from model_hybrid import model
from model_hybrid.train import flops_per_token
from torch.nn import functional as F

from tests.test_hybrid_backbone import tiny_config
from trainer.optimizers import parameter_groups


def looped_config(**overrides):
    settings = {
        "n_layer": 4,
        "layer_pattern": ("kda", "full"),
        "n_kda_head": 4,
        "loop_prelude": 1,
        "loop_coda": 1,
        "n_loops": 3,
    }
    return tiny_config(**(settings | overrides))


def explicit_trunk(lm, tokens, loops):
    """The looped forward pass written out block by block."""
    mask = model.DocumentMask(model.document_ids(tokens, lm.config.eos_id))
    rope = (lm.rope_cos[mask.positions][:, None], lm.rope_sin[mask.positions][:, None])
    prelude, first, second, coda = lm.blocks
    gate, rate = lm.loop_update.gate, lm.loop_update.rate
    injected = prelude(lm.token_embedding(tokens), rope, mask)
    state = injected
    for _ in range(loops):
        entry = state + injected
        proposal = second(first(entry, rope, mask), rope, mask)
        step = torch.sigmoid(entry @ gate.weight.T + gate.bias) * F.softplus(rate)
        state = state + step * (proposal - state)
    return coda(state, rope, mask)


class LoopedHybridTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_layout_and_validation(self):
        config = looped_config()
        self.assertTrue(config.looped)
        self.assertEqual(list(config.recurrent_blocks), [1, 2])
        self.assertEqual(config.block_uses(), (1, 3, 3, 1))
        self.assertEqual(config.block_uses(1), (1, 1, 1, 1))
        self.assertFalse(tiny_config().looped)
        self.assertEqual(tiny_config().block_uses(), (1, 1))
        for bad in (
            {"n_loops": 0},
            {"loop_prelude": -1},
            {"loop_prelude": 2, "loop_coda": 2},
            {"n_loops": 1},
        ):
            with self.assertRaises(ValueError, msg=bad):
                looped_config(**bad)

    def test_shares_weights_across_loops(self):
        looped = model.HybridLM(looped_config())
        plain = model.HybridLM(looped_config(loop_prelude=0, loop_coda=0, n_loops=1))
        width = looped.config.n_embd
        extra = sum(p.numel() for p in looped.loop_update.parameters())
        self.assertEqual(extra, width * width + 2 * width)
        self.assertEqual(
            sum(p.numel() for p in looped.parameters()),
            sum(p.numel() for p in plain.parameters()) + extra,
        )

    def test_gate_starts_at_a_uniform_step(self):
        update = model.HybridLM(looped_config()).loop_update
        self.assertTrue(all(not p.any() for p in update.parameters()))
        state, proposal = torch.zeros(1, 3, 32), torch.ones(1, 3, 32)
        stepped = update(state, proposal, torch.randn(1, 3, 32))
        torch.testing.assert_close(
            stepped, torch.full_like(stepped, 0.5 * F.softplus(torch.zeros(())))
        )

    def test_matches_explicit_loops_at_every_loop_count(self):
        lm = model.HybridLM(looped_config()).double()
        with torch.no_grad():
            for p in lm.loop_update.parameters():
                p.normal_(std=0.3)
        tokens = torch.randint(3, 37, (2, 13))
        tokens[0, 6] = lm.config.eos_id
        for loops in (1, 2, 3, 5):
            got = lm.trunk(tokens, loops)[0]
            torch.testing.assert_close(got, explicit_trunk(lm, tokens, loops))
        torch.testing.assert_close(lm.trunk(tokens)[0], explicit_trunk(lm, tokens, 3))
        self.assertFalse(torch.allclose(lm(tokens, loops=1), lm(tokens, loops=3)))
        with self.assertRaises(ValueError):
            lm(tokens, loops=0)
        with self.assertRaises(ValueError):
            model.HybridLM(tiny_config())(tokens, loops=2)

    def test_causal_and_document_isolated(self):
        lm = model.HybridLM(looped_config()).double()
        tokens = torch.randint(3, 37, (1, 12))
        changed = tokens.clone()
        changed[0, 8:] = torch.randint(3, 37, (4,))
        torch.testing.assert_close(lm(tokens)[:, :8], lm(changed)[:, :8])
        tokens[0, 5] = lm.config.eos_id
        alone = lm(tokens[:, 6:])
        torch.testing.assert_close(lm(tokens)[:, 6:], alone)

    def test_trains_and_every_parameter_gets_a_gradient(self):
        lm = model.HybridLM(looped_config())
        tokens = torch.randint(3, 37, (2, 12))
        labels = torch.roll(tokens, -1, 1)
        lm(tokens, labels)["loss"].backward()
        missing = [n for n, p in lm.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        groups = parameter_groups(lm)
        self.assertIn("loop_update.gate.weight", [name for name, _ in groups["muon"]])
        no_decay = [name for name, _ in groups["adamw_no_decay"]]
        self.assertIn("loop_update.rate", no_decay)
        self.assertIn("loop_update.gate.bias", no_decay)

    def test_flops_count_recurrent_blocks_once_per_loop(self):
        config = looped_config()
        lm = model.HybridLM(config)

        def weights(module):
            return sum(p.numel() for p in module.parameters() if p.ndim == 2)

        recurrent = sum(weights(lm.blocks[i]) for i in config.recurrent_blocks)
        gate = weights(lm.loop_update)
        matmul = sum(p.numel() for p in lm.parameters() if p.ndim == 2) + 2 * (recurrent + gate)
        full_uses = 3 + 1  # full blocks: 1 (recurrent, x3) and 3 (coda, x1)
        expected = 6 * matmul + 12 * full_uses * config.n_head * config.head_dim * 10
        self.assertEqual(flops_per_token(config, lm, 10), expected)


if __name__ == "__main__":
    unittest.main()
