import unittest

from distributed_training.config import DistributedConfig
from distributed_training.run_validation import validate_small_5b_run


class RunValidationTests(unittest.TestCase):
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
