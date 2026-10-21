import unittest

from distributed.config import DistributedConfig, DistributedRuntime, validate_resume


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
            {"pilot": "yes"},
            {"n_embd": 64},
            {"peak_lr": float("nan")},
            {"seed": -1},
            {"checkpoint_every": 0},
            {"target_tokens": 100, "warmup_tokens": 100},
            {"data_loader": "auto"},
            {"data_loader": None},
            {"adam_beta1": 1},
            {"adam_beta2": -0.1},
            {"adam_beta2": True},
            {"adam_beta1": float("nan")},
            {"gradient_clip_norm": 0},
            {"gradient_clip_norm": float("inf")},
            {"validation_windows_per_source": 0},
            {"validation_windows_per_source": True},
            {"log_every": 0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DistributedConfig(**change)

    def test_architecture_and_budget_are_not_experiment_scoped(self):
        config = DistributedConfig(n_layer=4, n_head=4, n_embd=256, target_tokens=30_000_000_000)
        self.assertEqual(config.n_layer, 4)
        # Run-budget policy belongs to execution validation, not construction.
        DistributedConfig(pilot=True, target_tokens=10_000_000, warmup_tokens=100_000)

    def test_unreviewed_data_is_an_explicit_boolean(self):
        self.assertFalse(DistributedConfig().allow_unreviewed_data)
        self.assertFalse(
            DistributedConfig(
                pilot=True, target_tokens=5_000_000, warmup_tokens=100_000
            ).allow_unreviewed_data
        )
        for value in ("yes", 1, 0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DistributedConfig(allow_unreviewed_data=value)
        self.assertTrue(DistributedConfig(allow_unreviewed_data=True).allow_unreviewed_data)

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

    def test_resume_rejects_changed_loader_optimizer_and_diagnostics(self):
        config = DistributedConfig()
        for change in (
            {"data_loader": "raw"},
            {"adam_beta1": 0.8},
            {"adam_beta2": 0.99},
            {"gradient_clip_norm": 0.5},
            {"validation_windows_per_source": 2},
            {"log_every": 20},
        ):
            saved = {
                "format": "scholarly-ddp-v1",
                "config": {**config.as_dict(), **change},
            }
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "identical training"),
            ):
                validate_resume(saved, config, {}, {}, "2.6")

    def test_runtime_resolves_torchrun_launch_details(self):
        runtime = DistributedRuntime.from_torchrun(
            {
                "RANK": "1",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "3",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29500",
            }
        )
        self.assertEqual(runtime, DistributedRuntime(1, 1, 3, "localhost", 29500))
        self.assertEqual(runtime.init_method, "tcp://localhost:29500")

    def test_runtime_rejects_incomplete_or_invalid_launch(self):
        with self.assertRaisesRegex(ValueError, "Missing torchrun launch variable: RANK"):
            DistributedRuntime.from_torchrun({})
        for fields in (
            {"rank": 3},
            {"local_rank": 3},
            {"local_rank": 1},
            {"world_size": 0},
            {"world_size": True},
            {"master_addr": ""},
            {"master_addr": "tcp://localhost"},
            {"master_port": 0},
            {"master_port": 65536},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                DistributedRuntime(
                    **{
                        "rank": 0,
                        "local_rank": 0,
                        "world_size": 3,
                        "master_addr": "localhost",
                        "master_port": 29500,
                        **fields,
                    }
                )

    def test_runtime_supports_explicit_ipv6_endpoint(self):
        runtime = DistributedRuntime(0, 0, 1, "::1", 29500)
        self.assertEqual(runtime.init_method, "tcp://[::1]:29500")


if __name__ == "__main__":
    unittest.main()
