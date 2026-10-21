import unittest
from unittest.mock import patch

from trainer.distributed_config import DistributedConfig
from trainer.distributed_train import train
from trainer.run_validation import validate_small_5b_run


class RunValidationTests(unittest.TestCase):
    def test_large_pilot_reaches_runtime_validation_without_a_token_ceiling(self):
        config = DistributedConfig(
            pilot=True, target_tokens=30_000_000_000, allow_unreviewed_data=True
        )
        # Deliberately reject a mismatched runtime before touching CUDA or files.
        with (
            patch.dict("os.environ", {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}),
            self.assertRaisesRegex(ValueError, "torchrun world size differs"),
        ):
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
