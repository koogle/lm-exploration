import unittest

from model_hybrid.benchmark import validate_window


class BenchmarkWindowTests(unittest.TestCase):
    def test_smoke_budget_cannot_qualify_twelve_steps(self):
        config = {"target_tokens": 8192, "model": {"block_size": 512}}
        with self.assertRaisesRegex(ValueError, "token budget"):
            validate_window(config, rows=2, steps=12, profile=True)

    def test_capture_requires_complete_window(self):
        config = {"target_tokens": 1000000, "model": {"block_size": 512}}
        with self.assertRaisesRegex(ValueError, "full capture"):
            validate_window(config, rows=2, steps=8, profile=True)
        validate_window(config, rows=2, steps=12, profile=True)
