"""A smaller final microbatch preserves global token-weighted NTP and MTP gradients."""

import contextlib
import copy
import dataclasses
import unittest
from unittest.mock import patch

import torch
from model_hybrid.model import HybridLM
from model_hybrid.train import HybridFamily, RunConfig

from tests.test_hybrid_backbone import tiny_config
from tests.test_training_optimizations import LocalDDP
from trainer.loop import accumulate


class PartialMicrobatchTests(unittest.TestCase):
    def test_two_rank_average_matches_unsplit_ntp_and_mtp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        config = tiny_config(mtp=True)
        with patch("model_hybrid.train.model_config_of", return_value=config):
            family = HybridFamily(
                RunConfig(recipe="", model=dataclasses.asdict(config), target_tokens=1000),
                ".",
                data=({}, None),
            )
        model = HybridLM(config)
        inputs = torch.randint(3, 37, (10, 8))
        inputs[:, 3] = 2
        labels = inputs.roll(-1, 1)
        labels[:3, -3:] = -100
        extras, counts = family.prepare(inputs, labels)
        progress = {"tokens": 0, "step": 0}
        objective, _ = family.loss(model, inputs, labels, extras, counts, progress)
        objective.backward()
        gradients = []
        for rank in range(2):
            replica = copy.deepcopy(model)
            replica.zero_grad(set_to_none=True)
            start = 5 * rank
            parts = list(
                zip(
                    inputs[start : start + 5].split(3),
                    labels[start : start + 5].split(3),
                    extras["mtp_labels"][start : start + 5].split(3),
                    strict=True,
                )
            )
            self.assertEqual([len(x) for x, _, _ in parts], [3, 2])
            ddp = LocalDDP(replica)

            def loss(x, y, mtp, ddp=ddp):
                return family.loss(ddp, x, y, {"mtp_labels": mtp}, counts, progress)

            with patch("torch.autocast", return_value=contextlib.nullcontext()):
                accumulate(ddp, parts, loss, world=2)
            gradients.append([p.grad.clone() for p in replica.parameters()])
        for parameter, a, b in zip(model.parameters(), *gradients, strict=True):
            torch.testing.assert_close((a + b) / 2, parameter.grad, atol=2e-6, rtol=2e-4)
