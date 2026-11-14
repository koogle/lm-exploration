import json
import unittest

from model_hybrid.benchmark import summarize, validate_window


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


class BenchmarkAggregateTests(unittest.TestCase):
    def test_stalls_count_in_aggregate_even_when_median_is_fast(self):
        rows = [
            json.dumps(
                {
                    "step": i,
                    "budget_tokens": 100,
                    "seconds": seconds,
                    "tokens_per_second": 100 / seconds,
                    "mfu": 0.3 / seconds,
                    "loss": 1.0,
                }
            )
            for i, seconds in enumerate((100, 1, 3, 1), start=1)
        ]
        result = summarize(rows, skip=1)
        self.assertEqual(result["tokens_per_second"], 100)
        self.assertEqual(result["aggregate_tokens_per_second"], 60)
        self.assertEqual(result["timed_seconds"], 5)
        self.assertEqual(result["timed_tokens"], 300)
        self.assertAlmostEqual(result["aggregate_mfu"], 0.18)

    def test_warmup_only_run_has_no_throughput_claim(self):
        row = json.dumps({"step": 1, "loss": 1.0})
        result = summarize([row])
        self.assertFalse(result["timing_valid"])
        self.assertIsNone(result["aggregate_tokens_per_second"])
        self.assertIsNone(result["aggregate_mfu"])
        self.assertIsNone(result["tokens_per_second"])
        self.assertIsNone(result["mfu"])
