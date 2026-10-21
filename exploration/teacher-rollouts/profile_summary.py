"""Summarize an explicit Chrome/Torch trace without treating summed time as wall time."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from core import atomic_json


def summarize_trace(events):
    kernels = defaultdict(lambda: {"calls": 0, "summed_device_us": 0.0})
    copies = defaultdict(lambda: {"calls": 0, "summed_device_us": 0.0})
    intervals = []
    for event in events:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        category = event.get("cat", "")
        if category == "kernel":
            bucket = kernels
        elif category in ("gpu_memcpy", "gpu_memset"):
            bucket = copies
        else:
            continue
        row = bucket[event["name"]]
        row["calls"] += 1
        row["summed_device_us"] += event["dur"]
        intervals.append((event["ts"], event["ts"] + event["dur"]))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0] if merged else 0

    def ranked(rows):
        return sorted(
            ({"name": name, **value} for name, value in rows.items()),
            key=lambda r: r["summed_device_us"],
            reverse=True,
        )

    return {
        "kernel_events": ranked(kernels),
        "explicit_copy_events": ranked(copies),
        "observed_device_interval_union_us": busy,
        "first_to_last_device_event_us": span,
        "observed_device_busy_fraction": busy / span if span else None,
        "limitations": [
            "Single trace only; do not combine independent device clocks.",
            "Summed kernel durations can overlap and are not wall-clock fractions.",
            "Kernel names do not establish bandwidth saturation or weight movement cost.",
            "Gaps can reflect idle requests, persistence, scheduling or profiling overhead.",
            "Prefill/decode and verifier costs require separate ranges or measurements.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as handle:
        trace = json.load(handle)
    report = summarize_trace(trace["traceEvents"])
    report["source_trace"] = str(args.trace)
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
