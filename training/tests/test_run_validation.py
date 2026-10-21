import unittest

from trainer.distributed_config import DistributedConfig
from trainer.distributed_train import train
from trainer.run_validation import validate_run, validate_small_5b_run


class RunValidationTests(unittest.TestCase):
    def test_generic_run_has_no_architecture_or_nonpilot_budget_allowlist(self):
        for allow_unreviewed in (False, True):
            config = DistributedConfig(
                n_layer=4,
                n_head=4,
                n_embd=256,
                target_tokens=30_000_000_000,
                allow_unreviewed_data=allow_unreviewed,
            )
            validate_run(config)

    def test_pilot_limit_is_a_run_policy(self):
        validate_run(
            DistributedConfig(
                pilot=True,
                target_tokens=5_000_000,
                warmup_tokens=100_000,
            )
        )
        config = DistributedConfig(
            pilot=True,
            target_tokens=5_000_001,
            warmup_tokens=100_000,
        )
        with self.assertRaisesRegex(ValueError, "Pilot runs"):
            validate_run(config)
        with self.assertRaisesRegex(ValueError, "Pilot runs"):
            train(config, "missing-manifest", "missing-output")

    def test_small_workflow_accepts_its_recipe_without_requiring_unreviewed_data(self):
        for allow_unreviewed in (False, True):
            validate_small_5b_run(
                DistributedConfig(
                    n_layer=8,
                    n_head=3,
                    n_embd=192,
                    microbatch=8,
                    target_tokens=5_000_000_000,
                    allow_unreviewed_data=allow_unreviewed,
                )
            )

    def test_small_workflow_rejects_other_runs(self):
        settings = {
            "n_layer": 8,
            "n_head": 3,
            "n_embd": 192,
            "microbatch": 8,
            "target_tokens": 5_000_000_000,
        }
        for change in (
            {"n_layer": 4},
            {"world_size": 2},
            {"target_tokens": 5_000_000_001},
            {"pilot": True, "target_tokens": 5_000_000, "warmup_tokens": 100_000},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_small_5b_run(DistributedConfig(**{**settings, **change}))


if __name__ == "__main__":
    unittest.main()
