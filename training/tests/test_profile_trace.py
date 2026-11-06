import unittest

from model_hybrid.profile_trace import launch_phase_times


class LaunchAttributionTests(unittest.TestCase):
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
