import unittest

from model_hybrid.profile_categories import kernel_category
from model_hybrid.profile_plot import kernel_durations


class ProfilePlotTests(unittest.TestCase):
    def test_custom_recurrence_kernels_are_not_other(self):
        for name in ("intra_backward", "wy_backward", "chunk_gla_fwd_kernel_o"):
            self.assertEqual(kernel_category(name), "kda_fla")
        self.assertEqual(kernel_category("kda_conv_backward"), "kda_convolution")

    def test_clips_kernel_to_disjoint_windows_preserving_overlap(self):
        trace = {
            "traceEvents": [
                {"ph": "X", "name": "ProfilerStep#1", "ts": 1000, "dur": 1000},
                {"ph": "X", "name": "ProfilerStep#2", "ts": 3000, "dur": 1000},
                {"ph": "X", "cat": "kernel", "name": "intra_backward", "ts": 500, "dur": 4000},
                {"ph": "X", "cat": "kernel", "name": "wy_backward", "ts": 1500, "dur": 1000},
            ]
        }
        rows = {r["name"]: r for r in kernel_durations(trace)}
        self.assertEqual(rows["intra_backward"]["ms_per_step"], 1)
        self.assertEqual(rows["wy_backward"]["ms_per_step"], 0.25)
        self.assertEqual(rows["wy_backward"]["calls_per_step"], 0.5)
