"""Muon/AdamW grouping for the hybrid family."""

import unittest

import torch
from model_hybrid import model, optim

from trainer.optimizers import set_learning_rate


def small_model():
    config = model.HybridConfig(
        vocab_size=50,
        eos_id=2,
        block_size=8,
        n_layer=4,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        n_kda_head=2,
        mlp_hidden=48,
    )
    return model.HybridLM(config)


class OptimizerGroupingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.lm = small_model()

    def test_every_parameter_lands_in_exactly_one_group(self):
        groups = optim.parameter_groups(self.lm)
        names = [name for members in groups.values() for name, _ in members]
        self.assertEqual(sorted(names), sorted(n for n, _ in self.lm.named_parameters()))
        self.assertEqual(len(names), len(set(names)))
        ids = [id(p) for members in groups.values() for _, p in members]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(1 for _ in self.lm.parameters()), len(ids))

        optimizer = optim.HybridOptimizer(self.lm)
        held = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(sorted(held), sorted(ids))

    def test_roles(self):
        roles = {n: r for r, members in optim.parameter_groups(self.lm).items() for n, _ in members}
        self.assertEqual(roles["token_embedding.weight"], "adamw_decay")
        self.assertEqual(roles["blocks.0.mixer.b_proj.weight"], "adamw_decay")
        for name in (
            "blocks.0.mixer.A_log",
            "blocks.0.mixer.dt_bias",
            "blocks.0.mixer.q_conv",
            "blocks.0.mixer.o_norm_weight",
            "blocks.3.mixer.sink",
            "blocks.3.mixer.gate_bias",
            "blocks.3.mixer.q_norm.weight",
            "final_norm.weight",
            "mtp.hidden_norm.weight",
        ):
            self.assertEqual(roles[name], "adamw_no_decay", name)
        for name in (
            "blocks.0.mixer.q_proj.weight",
            "blocks.0.mixer.f_proj.0.weight",
            "blocks.0.mixer.g_proj.1.weight",
            "blocks.3.mixer.gate_proj.weight",
            "blocks.3.mixer.o_proj.weight",
            "blocks.3.mlp.down.weight",
            "mtp.fuse.weight",
            "mtp.block.mixer.k_proj.weight",
        ):
            self.assertEqual(roles[name], "muon", name)
        muon = [p for n, p in self.lm.named_parameters() if roles[n] == "muon"]
        self.assertTrue(all(p.ndim == 2 for p in muon))

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            optim.optimizer_role("extra.weight", torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            optim.optimizer_role("blocks.0.odd", torch.zeros(2, 2, 2, 2))

    def test_learning_rates_and_step(self):
        optimizer = optim.HybridOptimizer(self.lm)
        set_learning_rate(optimizer, 0.5 * 3e-4)
        self.assertAlmostEqual(optimizer.muon.param_groups[0]["lr"], 0.01)
        decay, no_decay = optimizer.adamw.param_groups
        self.assertAlmostEqual(decay["lr"], 1.5e-4)
        self.assertEqual((decay["weight_decay"], no_decay["weight_decay"]), (0.1, 0.0))
        tokens = torch.randint(0, 50, (2, 8))
        before = {n: p.detach().clone() for n, p in self.lm.named_parameters()}
        losses = self.lm(tokens, tokens, mtp_labels=tokens)
        (losses["loss"] + losses["mtp_loss"]).backward()
        optimizer.step()
        changed = [n for n, p in self.lm.named_parameters() if not torch.equal(p, before[n])]
        self.assertEqual(len(changed), len(before))
        restored = optim.HybridOptimizer(small_model())
        restored.load_state_dict(optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
