import unittest

from model_hybrid.profile_trace import activity_union_ms, launch_phase_times, operator_kernel_times


class LaunchAttributionTests(unittest.TestCase):
    def test_matrix_work_counts_once_across_multiple_delayed_kernels(self):
        events = [
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "args": {
                    "External id": 7,
                    "Input Dims": [[2, 3], [3, 4]],
                    "Input type": ["float", "float"],
                },
            }
        ]
        kernels = [
            {"ts": 100, "dur": 20, "args": {"External id": 7}},
            {"ts": 200, "dur": 30, "args": {"External id": 7}},
            {"ts": 300, "dur": 5, "args": {"External id": 8}},
        ]
        rows = operator_kernel_times(events, kernels, 1)
        self.assertEqual(rows[0]["recognized_mm_flops_per_step"], 48)
        self.assertEqual(rows[0]["kernel_ms_per_step"], 0.05)
        self.assertEqual(rows[0]["operator_calls_per_step"], 1)
        self.assertEqual(rows[1]["operator"], "unattributed")

    def test_activity_union_clips_window_and_overlapping_streams(self):
        events = [{"ts": 0, "dur": 20}, {"ts": 15, "dur": 20}, {"ts": 50, "dur": 10}]
        self.assertEqual(activity_union_ms(events, 10, 55), 0.03)

    def test_delayed_kernel_stays_with_launch_not_later_host_phase(self):
        phases = [
            {"cat": "user_annotation", "ph": "X", "name": "train/accumulate", "ts": 0, "dur": 100},
            {"cat": "user_annotation", "ph": "X", "name": "train/backward", "ts": 20, "dur": 50},
            {"cat": "user_annotation", "ph": "X", "name": "train/clip", "ts": 100, "dur": 100},
            {"cat": "cuda_runtime", "ts": 40, "args": {"correlation": 1}},
        ]
        kernels = [
            {"ts": 110, "dur": 20, "args": {"correlation": 1}},
            {"ts": 130, "dur": 5, "args": {"correlation": 2}},
        ]
        self.assertEqual(
            launch_phase_times(phases, kernels, 1),
            {
                "train/backward": 0.02,
                "unattributed": 0.005,
            },
        )
