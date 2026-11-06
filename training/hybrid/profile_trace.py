"""Audit a benchmark's compressed Chrome trace without allocating a GPU.

    python -m model_hybrid.profile_trace --trace /path/trace.json.gz --output /path/audit.json

Host ranges are clipped to profiled steps. Device activity is a union, avoiding
counting overlapping streams twice. These instrumented times are diagnostic;
use unprofiled same-allocation runs for speed comparisons.
"""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path


def summarize(trace):
    events = trace["traceEvents"]
    steps = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("ProfilerStep")
        and e.get("ph") == "X"
    ]
    n = len(steps)
    lo = min(e["ts"] for e in steps)
    hi = max(e["ts"] + e["dur"] for e in steps)
    kernels = [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
    ]
    intervals = sorted(
        (max(lo, e["ts"]), min(hi, e["ts"] + e["dur"]))
        for e in kernels
        if e["ts"] < hi and e["ts"] + e["dur"] > lo
    )
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    ends = [x[1] for x in merged]

    def busy(a, b):
        i = bisect.bisect_right(ends, a)
        total = 0
        while i < len(merged) and merged[i][0] < b:
            total += max(0, min(b, merged[i][1]) - max(a, merged[i][0]))
            i += 1
        return total

    phases = collections.defaultdict(
        lambda: {"host_ms_per_step": 0, "gpu_busy_during_host_phase_ms_per_step": 0}
    )
    for e in events:
        if (
            e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("train/")
            and e.get("ph") == "X"
        ):
            a, b = max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])
            if b <= a:
                continue
            r = phases[e["name"]]
            r["host_ms_per_step"] += (b - a) / 1000 / n
            r["gpu_busy_during_host_phase_ms_per_step"] += busy(a, b) / 1000 / n
    ops = {}
    for name in [
        "aten::nonzero",
        "aten::repeat_interleave",
        "cudaStreamSynchronize",
        "cudaLaunchKernel",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "Command Buffer Full",
    ]:
        selected = [
            e
            for e in events
            if e.get("name") == name
            and e.get("ph") == "X"
            and e.get("cat") != "gpu_user_annotation"
        ]
        ops[name] = {
            "calls_per_step": len(selected) / n,
            "inclusive_host_ms_per_step": sum(e["dur"] for e in selected) / 1000 / n,
        }
    grouped = collections.defaultdict(list)
    for event in kernels:
        if event.get("cat") == "kernel":
            grouped[event["name"]].append(event)
    resources = []
    for name, calls in sorted(
        grouped.items(), key=lambda item: sum(e["dur"] for e in item[1]), reverse=True
    )[:35]:
        resources.append(
            {
                "name": name,
                "ms_per_step": sum(e["dur"] for e in calls) / 1000 / n,
                "calls_per_step": len(calls) / n,
                "registers_per_thread": sorted(
                    {
                        e["args"]["registers per thread"]
                        for e in calls
                        if "registers per thread" in e.get("args", {})
                    }
                ),
                "shared_bytes": sorted(
                    {
                        e["args"]["shared memory"]
                        for e in calls
                        if "shared memory" in e.get("args", {})
                    }
                ),
                "example_launch": {
                    k: calls[0].get("args", {}).get(k) for k in ("grid", "block", "occupancy")
                },
            }
        )
    r = {
        "kernel_resources": resources,
        "steps": n,
        "window_ms_per_step": (hi - lo) / 1000 / n,
        "cuda_activity_union_ms_per_step": busy(lo, hi) / 1000 / n,
        "observed_gpu_idle_fraction": 1 - busy(lo, hi) / (hi - lo),
        "phases": dict(phases),
        "operations": ops,
        "scope": "Profiler-instrumented rank 0. Host phases overlap; GPU overlap is activity during each host phase, not exclusive kernel attribution. Command-buffer overhead can be induced by profiling.",
    }
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with gzip.open(args.trace) as handle:
        result = summarize(json.load(handle))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
