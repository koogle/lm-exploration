import unittest

from profile_summary import summarize_trace


class ProfileTests(unittest.TestCase):
    def test_overlapping_kernels_copy_and_cpu_ranges(self):
        report = summarize_trace(
            [
                {"ph": "X", "cat": cat, "name": name, "ts": start, "dur": duration}
                for cat, name, start, duration in [
                    ("kernel", "gemm", 0, 10),
                    ("kernel", "delta", 5, 10),
                    ("gpu_memcpy", "HtoD", 20, 5),
                    ("cpu_op", "model.forward", 0, 100),
                ]
            ]
        )
        self.assertEqual(report["observed_device_interval_union_us"], 20)
        self.assertEqual(report["first_to_last_device_event_us"], 25)
        self.assertEqual(report["observed_device_busy_fraction"], 0.8)
        self.assertEqual(len(report["kernel_events"]), 2)
        self.assertEqual(report["explicit_copy_events"][0]["summed_device_us"], 5)

    def test_empty_profile_does_not_claim_zero_utilization(self):
        self.assertIsNone(summarize_trace([])["observed_device_busy_fraction"])
