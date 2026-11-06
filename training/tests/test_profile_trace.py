import unittest

from model_hybrid.profile_trace import (
    activity_union_ms,
    launch_phase_times,
    operator_kernel_times,
    summarize,
)


class LaunchAttributionTests(unittest.TestCase):
    def test_host_wait_is_not_counted_as_device_idle(self):
        def event(category, name, start, duration):
            return {"cat": category, "name": name, "ph": "X", "ts": start, "dur": duration}

        trace = {
            "traceEvents": [
                event("user_annotation", "ProfilerStep#1", 0, 100000),
                event("kernel", "matrix_work", 0, 75000),
                event("cpu_op", "aten::nonzero", 20000, 60000),
                event("cpu_op", "aten::repeat_interleave", 82000, 8000),
                event("cpu_op", "aten::repeat_interleave", 83000, 6000),
            ]
        }
        ops = summarize(trace)["operations"]
        self.assertEqual(ops["aten::nonzero"]["inclusive_host_ms_per_step"], 60)
        self.assertEqual(ops["aten::nonzero"]["device_idle_during_host_windows_ms_per_step"], 5)
        self.assertEqual(ops["aten::repeat_interleave"]["host_window_union_ms_per_step"], 8)

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


class ExclusiveLedgerTests(unittest.TestCase):
    @staticmethod
    def event(cat, name, start, duration, **extra):
        return dict(cat=cat, name=name, ts=start, dur=duration, ph="X", **extra)

    def test_partition_overlaps_clipping_and_empty_gaps(self):
        from model_hybrid.profile_trace import exclusive_ledger

        events = [
            self.event("kernel", "gemm", -10, 30),
            self.event("kernel", "gemm2", 10, 30),
            self.event("kernel", "ncclAllReduce", 30, 30),
            self.event("gpu_memcpy", "copy", 50, 20),
            self.event("gpu_memset", "zero", 65, 10),
            self.event("cpu_op", "wait", 0, 100),
            self.event("kernel", "outside", 100, 20),
        ]
        result = exclusive_ledger(events, 0, 100)
        self.assertEqual(
            result["exclusive_ms"],
            {
                "compute_only": 0.03,
                "communication_only": 0.01,
                "memops_only": 0.015,
                "overlap": 0.02,
                "idle": 0.025,
            },
        )
        self.assertAlmostEqual(result["residual_ms"], 0)
        self.assertEqual(exclusive_ledger([], 0, 100)["exclusive_ms"]["idle"], 0.1)

    def test_attribution_and_step_clipping(self):
        from model_hybrid.profile_trace import timeline_data

        events = [
            self.event("user_annotation", "ProfilerStep#1", 0, 100),
            self.event("user_annotation", "ProfilerStep#2", 200, 100),
            self.event("cpu_op", "aten::mm", 0, 10, args={"External id": 5}),
            self.event("kernel", "gemm", 90, 130, args={"External id": 5}),
            self.event("kernel", "unknown", 250, 10),
        ]
        result = timeline_data({"traceEvents": events}, "7")
        self.assertEqual(result["rank"], "7")
        self.assertEqual(result["steps"][0]["kernel_totals"], [["aten::mm", "gemm", 0.01]])
        self.assertEqual(result["steps"][1]["unattributed_kernel_sum_ms"], 0.01)
        for step in result["steps"]:
            self.assertAlmostEqual(sum(step["exclusive_ms"].values()), 0.1)

    def test_html_escapes_embedded_script_and_missing_steps(self):
        import tempfile
        from pathlib import Path

        from model_hybrid.profile_trace import render_timeline, timeline_data

        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "trace.html"
            render_timeline([{"rank": "</script><script>alert(1)</script>"}], p)
            self.assertNotIn('"</script>', p.read_text())
            import base64
            import gzip

            encoded = (
                p.read_text().split('type="application/octet-stream">')[1].split("</script>")[0]
            )
            self.assertIn("\\u003c/script", gzip.decompress(base64.b64decode(encoded)).decode())
        with self.assertRaises(ValueError):
            timeline_data({"traceEvents": []}, "0")


class HostDetailTests(unittest.TestCase):
    def test_bounded_host_display_preserves_totals_and_stack(self):
        from model_hybrid.profile_trace import host_details

        events = [
            {
                "cat": "cpu_op",
                "ph": "X",
                "name": "outer",
                "ts": -10,
                "dur": 100,
                "tid": 1,
                "args": {"Call stack": "a.py:2"},
            },
            {
                "cat": "cuda_runtime",
                "ph": "X",
                "name": "launch",
                "ts": 10,
                "dur": 5,
                "tid": 1,
                "args": {"correlation": 9},
            },
        ]
        result = host_details(events, 0, 50, limit=1)
        self.assertEqual(result["omitted_spans"], 1)
        self.assertEqual(
            result["totals"], [["cpu_op", "outer", 0.05], ["cuda_runtime", "launch", 0.005]]
        )
        self.assertIn("a.py:2", result["spans"][0][4])
        self.assertEqual(len(result["launches"]), 1)

    def test_kernel_launch_and_idle_neighbors_are_explicit(self):
        import json

        from model_hybrid.profile_trace import timeline_data

        events = [
            {"cat": "user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 0, "dur": 100},
            {
                "cat": "cuda_runtime",
                "ph": "X",
                "name": "launch",
                "ts": 10,
                "dur": 3,
                "tid": 4,
                "args": {"correlation": 9},
            },
            {
                "cat": "kernel",
                "ph": "X",
                "name": "gemm",
                "ts": 20,
                "dur": 20,
                "args": {"correlation": 9},
            },
        ]
        result = timeline_data({"traceEvents": events, "baseTimeNanoseconds": 123}, "2")
        step = result["steps"][0]
        self.assertEqual(json.loads(step["spans"][0][5])["launch_start_ms"], 0.01)
        self.assertEqual(step["idle_gap_context"][-1]["preceding_launch"][3], "launch")
        self.assertEqual(result["clock_metadata"]["baseTimeNanoseconds"], 123)
