import unittest

from trainer.distributed_config import DistributedConfig, validate_resume


class DistributedConfigTests(unittest.TestCase):
    def test_resume_rejects_optimizer_switch(self):
        config = DistributedConfig()
        self.assertEqual(config.optimizer, "muon")
        saved = {
            "format": "scholarly-ddp-v1",
            "config": DistributedConfig(optimizer="adamw").as_dict(),
        }
        with self.assertRaisesRegex(ValueError, "identical training settings"):
            validate_resume(saved, config, {}, {}, "2.6")

    def test_matched_batch_sizes(self):
        for world, batch in ((1, 24), (2, 24), (3, 16)):
            config = DistributedConfig(world_size=world, microbatch=batch)
            self.assertEqual(config.global_windows, 48)

    def test_invalid_configs(self):
        for change in (
            {"world_size": 4},
            {"prefetch_batches": True},
            {"prefetch_batches": 65},
            {"prefetch_batches": -1},
            {"world_size": 2, "microbatch": 16},
            {"target_tokens": True},
            {"pilot": "yes"},
            {"n_embd": 64},
            {"peak_lr": float("nan")},
            {"seed": -1},
            {"checkpoint_every": 0},
            {"target_tokens": 100, "warmup_tokens": 100},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DistributedConfig(**change)

    def test_architecture_and_budget_are_not_experiment_scoped(self):
        for settings in (
            {"n_layer": 8, "n_head": 3, "n_embd": 192, "target_tokens": 5_000_000_000},
            {"n_layer": 4, "n_head": 4, "n_embd": 256, "target_tokens": 30_000_000_000},
            {"pilot": True, "target_tokens": 10_000_000},
        ):
            with self.subTest(settings=settings):
                config = DistributedConfig(**settings, warmup_tokens=100_000)
                self.assertEqual(config.target_tokens, settings["target_tokens"])

    def test_unreviewed_data_requires_explicit_boolean(self):
        self.assertFalse(DistributedConfig().allow_unreviewed_data)
        self.assertFalse(
            DistributedConfig(
                pilot=True, target_tokens=5_000_000, warmup_tokens=100_000
            ).allow_unreviewed_data
        )
        for value in ("yes", 1, 0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DistributedConfig(allow_unreviewed_data=value)

    def test_unreviewed_data_is_independent_of_run_shape(self):
        for settings in (
            {},
            {"n_layer": 8, "n_head": 3, "n_embd": 192, "target_tokens": 5_000_000_001},
            {"pilot": True, "target_tokens": 5_000_000, "warmup_tokens": 100_000},
        ):
            with self.subTest(settings=settings):
                config = DistributedConfig(allow_unreviewed_data=True, **settings)
                self.assertTrue(config.allow_unreviewed_data)
                self.assertTrue(config.as_dict()["allow_unreviewed_data"])

    def test_authorized_eight_gpu_124m_run(self):
        config = DistributedConfig(
            world_size=8,
            global_windows=32,
            microbatch=1,
            context=8192,
            target_tokens=5_000_000_000,
            allow_unreviewed_data=True,
            prefetch_batches=4,
        )
        self.assertEqual(config.global_windows // config.world_size, 4)
        self.assertEqual(config.context, 8192)

    def test_schedule_uses_tokens(self):
        config = DistributedConfig()
        self.assertAlmostEqual(config.learning_rate(config.warmup_tokens), config.peak_lr)
        self.assertAlmostEqual(
            config.learning_rate(config.target_tokens), config.peak_lr * config.min_lr_ratio
        )
        self.assertLess(config.learning_rate(config.warmup_tokens // 2), config.peak_lr)

    def test_resume_rejects_drift(self):
        config = DistributedConfig()
        saved = {
            "format": "scholarly-ddp-v1",
            "config": config.as_dict(),
            "data_identity": {"hash": "x"},
            "source_sha256": {"model": "y"},
            "torch_version": "2.6",
        }
        validate_resume(saved, config, {"hash": "x"}, {"model": "y"}, "2.6")
        for field, value in (
            ("format", "old"),
            ("config", {}),
            ("data_identity", {}),
            ("source_sha256", {}),
            ("torch_version", "other"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume(
                    {**saved, field: value}, config, {"hash": "x"}, {"model": "y"}, "2.6"
                )


if __name__ == "__main__":
    unittest.main()
