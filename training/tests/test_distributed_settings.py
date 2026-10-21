"""Check configured optimizer behavior and fixed-window diagnostic sampling."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from distributed.config import DistributedConfig
from trainer.loop import evaluate_sources, make_optimizer, validation_blocks
from workflows.config import load_workflow_config


class DistributedSettingsTests(unittest.TestCase):
    def test_optimizer_uses_nondefault_recipe_after_json_roundtrip(self):
        config = DistributedConfig(
            adam_beta1=0.8,
            adam_beta2=0.99,
            peak_lr=0.002,
            weight_decay=0.02,
            gradient_clip_norm=0.25,
            validation_windows_per_source=3,
            log_every=7,
        )
        restored = DistributedConfig(**json.loads(json.dumps(config.as_dict())))
        self.assertEqual(restored, config)
        parameter = torch.nn.Linear(2, 1)
        optimizer = make_optimizer(parameter, restored, fused=False)
        group = optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.8, 0.99))
        self.assertEqual(group["lr"], 0.002)
        self.assertEqual(group["weight_decay"], 0.02)
        parameter(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertTrue(optimizer.state)

    def test_validation_samples_exact_count_and_fails_if_split_is_too_small(self):
        data = SimpleNamespace(sources={"source": {}}, blocks=lambda _: (i for i in range(5)))
        self.assertEqual(validation_blocks(data, 3), {"source": [0, 1, 2]})
        self.assertEqual(validation_blocks(data, 1), {"source": [0]})
        with self.assertRaisesRegex(ValueError, "fewer than 6"):
            validation_blocks(data, 6)

    def test_validation_weights_unequal_windows_by_valid_targets(self):
        with torch.random.fork_rng():
            torch.manual_seed(12)
            model = torch.nn.Embedding(8, 8)
            blocks = [
                {"input_ids": [1, 2, 3], "labels": [2, 3, 4], "loss_tokens": 3},
                {"input_ids": [5, 0, 0], "labels": [6, -100, -100], "loss_tokens": 1},
            ]
            loss = evaluate_sources(model, {"source": blocks}, device="cpu", bf16=False)["source"]
            expected = F.cross_entropy(
                model(torch.tensor([1, 2, 3, 5])), torch.tensor([2, 3, 4, 6])
            )
            self.assertAlmostEqual(loss, expected.item(), places=6)

    def test_workflow_policy_rejects_invalid_settings(self):
        for change in (
            {"world_size": 4},
            {"source_workers": 0},
            {"database_batch_size": True},
            {"dataset_config": "../recipe.json"},
            {"model_config": "/recipe.json"},
            {"segment_seconds": 90000},
            {"worker_timeout_seconds": 21600},
            {"segments_per_call": 4},
            {"qualification_segments": 20},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(
                    load_workflow_config(
                        Path(__file__).resolve().parents[1] / "workflows/small-5b.json"
                    ),
                    **change,
                )
