import unittest
from pathlib import Path

from distributed_training.config import DistributedConfig, DistributedRuntime
from distributed_training.run_validation import validate_run
from distributed_training.train import train


class RunValidationTests(unittest.TestCase):
    def test_nonpilot_architecture_and_budget_are_recipe_choices(self):
        for allow_unreviewed in (False, True):
            config = DistributedConfig(
                n_layer=4,
                n_head=4,
                n_embd=256,
                target_tokens=30_000_000_000,
                allow_unreviewed_data=allow_unreviewed,
            )
            validate_run(config)

    def test_pilot_policy_boundary(self):
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

    def test_direct_training_rejects_before_framework_or_filesystem_setup(self):
        config = DistributedConfig(
            pilot=True,
            target_tokens=5_000_001,
            warmup_tokens=100_000,
            allow_unreviewed_data=True,
        )
        with self.assertRaisesRegex(ValueError, "Pilot runs"):
            train(
                config,
                Path("missing-manifest.json"),
                Path("missing-output"),
                runtime=DistributedRuntime(0, 0, 3, "localhost", 29500),
            )


if __name__ == "__main__":
    unittest.main()
