"""Consolidation must preserve masked objectives, tied gradients and parameter identity."""

import copy
import unittest

import torch
from model_hybrid import model
from model_hybrid.consolidation import configure
from model_hybrid.fused_loss import fused_cross_entropy, joint_cross_entropy


class ConsolidationTests(unittest.TestCase):
    def test_joint_objective_unequal_heads_masks_and_upstream_scale(self):
        torch.manual_seed(8)
        for mtp_scale in (0.0, 0.3 / 5):
            first = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
            second = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
            weight = torch.randn(37, 16, dtype=torch.float64, requires_grad=True)
            y = torch.randint(0, 37, (2, 7))
            y[0, 3:] = -100
            future = torch.tensor([4, -100, 9, -100, 12])
            scales = (
                torch.tensor(1 / 10, dtype=torch.float64),
                torch.tensor(mtp_scale, dtype=torch.float64),
            )
            joint = joint_cross_entropy(first, second, weight, y, future, scales, 1e-4, 3)
            a = fused_cross_entropy(first, weight, y, 1e-4, 3)
            b = fused_cross_entropy(second, weight, future, 1e-4, 3)
            expected = scales[0] * a[0] + scales[1] * b[0]
            torch.testing.assert_close(joint[0], expected)
            for actual, reference in zip(joint[1:], (*a[1:], *b[1:]), strict=True):
                torch.testing.assert_close(actual, reference)
            actual = torch.autograd.grad(8 * joint[0], (first, second, weight))
            reference = torch.autograd.grad(8 * expected, (first, second, weight))
            for got, want in zip(actual, reference, strict=True):
                torch.testing.assert_close(got, want)

    def test_grouped_projections_and_joint_loss_keep_tied_parameter_gradients(self):
        torch.manual_seed(9)
        config = model.HybridConfig(
            vocab_size=37,
            eos_id=2,
            block_size=8,
            n_layer=2,
            n_embd=32,
            n_head=2,
            n_kv_head=1,
            head_dim=16,
            n_kda_head=2,
            mlp_hidden=64,
            layer_pattern=("kda", "full"),
            mtp=True,
        )
        original = model.HybridLM(config).double()
        candidate = copy.deepcopy(original)
        parameters = dict(candidate.named_parameters())
        configure(candidate, "projections")
        self.assertEqual(list(parameters), list(dict(candidate.named_parameters())))
        self.assertTrue(
            all(p is dict(candidate.named_parameters())[n] for n, p in parameters.items())
        )
        tokens = torch.randint(3, 37, (2, 7))
        labels = torch.randint(3, 37, (2, 7))
        labels[0, 3:] = -100
        mtp = labels.roll(-1, 1)
        mtp[:, -1] = -100
        scales = (torch.tensor(0.1, dtype=torch.float64), torch.tensor(0.03, dtype=torch.float64))
        reference = original(tokens, labels, mtp)
        objective = scales[0] * reference["loss"] + scales[1] * reference["mtp_loss"]
        actual = candidate(tokens, labels, mtp, loss_scales=scales)["objective"]
        torch.testing.assert_close(actual, objective)
        objective.backward()
        actual.backward()
        for (name, p), (_, q) in zip(
            original.named_parameters(), candidate.named_parameters(), strict=True
        ):
            torch.testing.assert_close(q.grad, p.grad, msg=name)
