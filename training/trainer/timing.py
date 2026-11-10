"""Optional step-phase timing without extra device synchronization.

CUDA durations are elapsed time on the training stream, including its waits and
host submission gaps. Nested phases overlap; these are not kernel busy times.
"""

import contextlib
import time
from collections import defaultdict

import torch


class StepTiming:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.samples = []

    @contextlib.contextmanager
    def phase(self, name):
        with torch.profiler.record_function(f"train/{name}"):
            if not self.enabled:
                yield
                return
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            tick = time.perf_counter()
            try:
                yield
            finally:
                end.record()
                self.samples.append((name, start, end, 1000 * (time.perf_counter() - tick)))

    def result(self):
        """Read only after the caller's existing end-of-step synchronization."""
        totals = defaultdict(lambda: {"host_ms": 0.0, "cuda_stream_ms": 0.0})
        for name, start, end, host_ms in self.samples:
            totals[name]["host_ms"] += host_ms
            totals[name]["cuda_stream_ms"] += start.elapsed_time(end)
        return dict(totals)
