"""The 720M hybrid run config, its schedule and accounting (CPU)."""

import dataclasses
import unittest
from pathlib import Path

import numpy as np
import torch
from model_hybrid import model, tokenizer
from model_hybrid.benchmark import SyntheticMixture, summarize
from model_hybrid.train import RunConfig, flops_per_token, learning_rate_factor, model_config_of

CONFIG = Path(__file__).parents[1] / "hybrid" / "configs" / "hybrid-720m-40b.json"


class HybridRunConfigTests(unittest.TestCase):
    def setUp(self):
        self.run = RunConfig.load(CONFIG)
        self.config = model_config_of(self.run)
        with torch.device("meta"):
            self.lm = model.HybridLM(self.config)

    def test_architecture_and_parameter_count(self):
        config = self.config
        self.assertEqual(
            (config.n_embd, config.n_layer, config.n_head, config.n_kv_head), (1536, 16, 12, 4)
        )
        self.assertEqual(
            (config.n_kda_head, config.head_dim, config.vocab_size), (12, 128, tokenizer.VOCAB_SIZE)
        )
        self.assertEqual(
            [i for i, k in enumerate(config.layer_kinds) if k == "full"], [3, 7, 11, 15]
        )
        total = sum(p.numel() for p in self.lm.parameters())
        trunk = total - sum(p.numel() for p in self.lm.mtp.parameters())
        self.assertEqual(trunk, 707_129_536)
        self.assertEqual(total, 739_972_556)
        for count in (trunk, total):
            self.assertTrue(0.70e9 <= count <= 0.75e9)

    def test_batch_and_schedule(self):
        run = self.run
        self.assertEqual(run.global_rows * self.config.block_size, 1_048_576)
        self.assertEqual((run.gradient_clip, run.optimizer["weight_decay"]), (1.0, 0.1))
        self.assertLessEqual(run.checkpoint_seconds, 2 * 3600)
        self.assertEqual(learning_rate_factor(run, 0), 0)
        self.assertAlmostEqual(learning_rate_factor(run, run.warmup_tokens / 2), 0.5)
        self.assertEqual(learning_rate_factor(run, 0.5 * run.target_tokens), 1.0)
        self.assertEqual(learning_rate_factor(run, 0.85 * run.target_tokens), 1.0)
        self.assertAlmostEqual(learning_rate_factor(run, 0.925 * run.target_tokens), 0.5)
        self.assertEqual(learning_rate_factor(run, run.target_tokens), 0.0)

    def test_flops_per_token(self):
        matmul = sum(p.numel() for p in self.lm.parameters() if p.ndim == 2)
        expected = 6 * (matmul + 154_880 * 1536) + 12 * 5 * 12 * 128 * 1000
        self.assertEqual(flops_per_token(self.config, self.lm, 1000), expected)

    def test_synthetic_benchmark_data_and_summary(self):
        mixture = SyntheticMixture(64, tokenizer.VOCAB_SIZE, 16)
        inputs, labels, _ = mixture.next_batch(4)
        self.assertEqual(inputs.shape, (4, 64))
        np.testing.assert_array_equal(inputs[:, 1:], labels[:, :-1])
        self.assertGreater((inputs == tokenizer.EOS_ID).sum(), 0)
        lines = [
            f'{{"step": {i}, "tokens_per_second": {i}, "mfu": 0.3, "loss": 1}}' for i in range(1, 9)
        ]
        self.assertEqual(summarize(lines, skip=5)["tokens_per_second"], 7)

    def test_attention_kernel_choice_and_statistics(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.config, attention_kernel="other")
        small = model.HybridConfig(
            50,
            block_size=8,
            n_layer=1,
            n_embd=32,
            n_head=4,
            n_kv_head=2,
            head_dim=8,
            mlp_hidden=48,
            layer_pattern=("full",),
            mtp=False,
        )
        lm = model.HybridLM(small)
        model.Attention.record_statistics = True
        try:
            lm(torch.randint(0, 50, (2, 8)))
        finally:
            model.Attention.record_statistics = False
        stats = lm.blocks[0].mixer.statistics
        self.assertGreater(stats["gate_open"], 0.9)
        self.assertTrue(0 < stats["sink_weight"] < 1)


if __name__ == "__main__":
    unittest.main()
