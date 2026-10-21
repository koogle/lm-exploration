import unittest

from distributed_training.config import DistributedConfig, DistributedRuntime
from distributed_training.train import train


class RunValidationTests(unittest.TestCase):
    def test_large_pilot_reaches_runtime_validation_without_a_token_ceiling(self):
        config = DistributedConfig(
            pilot=True,
            target_tokens=30_000_000_000,
            allow_unreviewed_data=True,
        )
        # A mismatched launch must fail before CUDA setup or filesystem writes,
        # not because this recipe exceeds an experiment-specific token ceiling.
        with self.assertRaisesRegex(ValueError, "torchrun world size differs"):
            train(
                config,
                "missing-manifest",
                "missing-output",
                runtime=DistributedRuntime(0, 0, 1, "localhost", 29500),
            )


if __name__ == "__main__":
    unittest.main()
