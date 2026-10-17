import unittest

from trainer.distributed_config import DistributedConfig, validate_resume


class DistributedConfigTests(unittest.TestCase):
    def test_matched_batch_sizes(self):
        for world, batch in ((1, 24), (2, 24), (3, 16)):
            config = DistributedConfig(world_size=world, microbatch=batch)
            self.assertEqual(config.global_windows, 48)

    def test_invalid_configs(self):
        for change in (
            {"world_size": 4},
            {"world_size": 2, "microbatch": 16},
            {"target_tokens": True},
            {"pilot": True},
            {"n_embd": 64},
            {"peak_lr": float("nan")},
            {"seed": -1},
            {"checkpoint_every": 0},
            {"target_tokens": 100, "warmup_tokens": 100},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DistributedConfig(**change)

    def test_modern_small_architecture(self):
        config = DistributedConfig(n_layer=8, n_head=3, n_embd=192, target_tokens=5_000_000_000)
        self.assertFalse(config.pilot)

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
