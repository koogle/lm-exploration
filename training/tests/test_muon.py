import copy
import unittest

import torch
from model_100m.model import GPT, ModelConfig

from distributed.config import DistributedConfig
from trainer.optimizers import (
    make_optimizer,
    orthogonalize,
    parameter_groups,
    set_learning_rate,
)


class MuonTests(unittest.TestCase):
    def model(self):
        return GPT(
            ModelConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16, eos_id=2)
        )

    def test_partition_and_schedule(self):
        model = self.model()
        config = DistributedConfig()
        opt = make_optimizer(model, config, fused=False)
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertIn(
            id(model.token_embedding.weight),
            {id(p) for g in opt.adamw.param_groups for p in g["params"]},
        )
        set_learning_rate(opt, 0.5)
        self.assertAlmostEqual(opt.muon.param_groups[0]["lr"], config.muon_lr / 2)
        self.assertAlmostEqual(opt.adamw.param_groups[0]["lr"], config.peak_lr / 2)

    def test_resume_next_update_exact(self):
        torch.manual_seed(11)
        model = self.model()
        config = DistributedConfig()
        opt = make_optimizer(model, config, fused=False)
        x = torch.randint(0, 64, (2, 8))

        def update(m, o):
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()

        update(model, opt)
        restored = copy.deepcopy(model)
        resumed = make_optimizer(restored, config, fused=False)
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        set_learning_rate(opt, 1 / 3)
        set_learning_rate(resumed, 1 / 3)
        update(model, opt)
        update(restored, resumed)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_zero_and_rectangular_polar(self):
        for shape in ((4, 16), (16, 4), (8, 8)):
            self.assertEqual(orthogonalize(torch.zeros(shape)).count_nonzero(), 0)
            q = torch.linalg.qr(torch.randn(max(shape), min(shape)))[0]
            if shape[0] < shape[1]:
                q = q.T
            actual = orthogonalize(q).float()
            self.assertTrue(torch.isfinite(actual).all())
            # Quintic iteration intentionally approximates, rather than converges to, polar.
            self.assertLess(float((actual - q).norm() / q.norm()), 0.4)

    def test_explicit_adamw_matches_baseline(self):
        model = self.model()
        reference = copy.deepcopy(model)
        config = DistributedConfig(optimizer="adamw")
        optimizer = make_optimizer(model, config, fused=False)
        baseline = torch.optim.AdamW(
            reference.parameters(),
            lr=config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=False,
        )
        for a, b in zip(model.parameters(), reference.parameters()):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        optimizer.step()
        baseline.step()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_gpt_roles_keep_matrices_on_muon_and_the_rest_on_decayed_adamw(self):
        model = self.model()
        groups = parameter_groups(model)
        self.assertEqual(groups["adamw_no_decay"], [])
        muon = {n for n, _ in groups["muon"]}
        expected = {
            n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2
        }
        self.assertEqual(muon, expected)

    def test_invalid_settings(self):
        for values in (
            {"optimizer": "sgd"},
            {"muon_lr": 0},
            {"muon_lr": float("nan")},
            {"muon_momentum": 1},
        ):
            with self.assertRaises(ValueError):
                DistributedConfig(**values)
