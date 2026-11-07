"""Keep the warmed training object graph out of automatic cyclic-GC scans."""

import gc
import os
import time
from pathlib import Path


def host_rss_bytes():
    """Current resident memory on Linux (not the process lifetime high-water mark)."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class TrainingGC:
    def __init__(self, warmup_steps=0):
        if warmup_steps < 0:
            raise ValueError("GC warmup steps must be non-negative")
        self.warmup_steps = warmup_steps
        self.frozen = False

    def after_step(self, session_step):
        if not self.warmup_steps or self.frozen or session_step != self.warmup_steps:
            return None
        started = time.perf_counter()
        previous_frozen = gc.get_freeze_count()
        collected = gc.collect()
        gc.freeze()
        self.frozen = True
        return {
            "session_step": session_step,
            "previous_frozen_objects": previous_frozen,
            "collected": collected,
            "frozen_objects": gc.get_freeze_count(),
            "ms": (time.perf_counter() - started) * 1000,
            "host_rss_bytes": host_rss_bytes(),
        }

    def close(self):
        if self.frozen:
            gc.unfreeze()
            self.frozen = False
