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


def operator_kernel_times(events, kernels, steps):
    """Link device work to the originating operator's shapes, not its execution time.

    FLOPs cover recognized two-dimensional aten::mm calls only; custom GEMMs,
    attention, optimizer work and recurrent kernels are not inferred from names.
    Each operator's FLOPs count once even when it launches several kernels.
    """
    operators = {
        e["args"]["External id"]: e
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    grouped = {}
    for kernel in kernels:
        identity = kernel.get("args", {}).get("External id")
        op = operators.get(identity, {})
        args = op.get("args", {})
        shapes, dtypes = args.get("Input Dims", []), args.get("Input type", [])
        name = op.get("name", "unattributed")
        key = (name, json.dumps(shapes), json.dumps(dtypes))
        if key not in grouped:
            grouped[key] = {
                "operator": name,
                "input_shapes": shapes,
                "input_types": dtypes,
                "kernel_ms_per_step": 0.0,
                "kernel_calls_per_step": 0.0,
                "recognized_mm_flops_per_step": 0.0,
                "operators": set(),
            }
        row = grouped[key]
        row["kernel_ms_per_step"] += kernel["dur"] / 1000 / steps
        row["kernel_calls_per_step"] += 1 / steps
        if identity not in row["operators"]:
            row["operators"].add(identity)
            if name == "aten::mm" and len(shapes) >= 2:
                a, b = shapes[:2]
                if len(a) == len(b) == 2 and a[1] == b[0]:
                    row["recognized_mm_flops_per_step"] += 2 * a[0] * a[1] * b[1] / steps
    for row in grouped.values():
        row["operator_calls_per_step"] = len(row.pop("operators")) / steps
    return sorted(grouped.values(), key=lambda r: r["kernel_ms_per_step"], reverse=True)


def activity_union_ms(events, lo, hi):
    """Union within a window: overlaps on separate streams count once."""
    intervals = sorted(
        (max(lo, e["ts"]), min(hi, e["ts"] + e["dur"]))
        for e in events
        if e["ts"] < hi and e["ts"] + e["dur"] > lo
    )
    total, end = 0.0, lo
    for a, b in intervals:
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total / 1000


def launch_phase_times(events, kernels, steps):
    """Attribute kernel duration to its CPU launch phase, not execution-time overlap.

    Backward kernels can execute after their host annotation ends. Correlation
    IDs link each device operation to its launch; nested phases count only once.
    Durations still sum across streams and are not exclusive wall-clock time.
    """
    phases = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("train/")
        and e.get("ph") == "X"
    ]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    totals = collections.Counter()
    for event in kernels:
        launch = launches.get(event.get("args", {}).get("correlation"))
        matches = [p for p in phases if launch and p["ts"] <= launch["ts"] < p["ts"] + p["dur"]]
        name = min(matches, key=lambda p: p["dur"])["name"] if matches else "unattributed"
        totals[name] += event["dur"] / 1000 / steps
    return dict(totals)


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
        windows = []
        for a, b in sorted(
            (max(lo, e["ts"]), min(hi, e["ts"] + e["dur"]))
            for e in selected
            if e["ts"] < hi and e["ts"] + e["dur"] > lo
        ):
            if windows and a <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], b)
            else:
                windows.append([a, b])
        host_union = sum(b - a for a, b in windows) / 1000 / n
        ops[name] = {
            "calls_per_step": len(selected) / n,
            "inclusive_host_ms_per_step": sum(e["dur"] for e in selected) / 1000 / n,
            "host_window_union_ms_per_step": host_union,
            "device_idle_during_host_windows_ms_per_step": host_union
            - sum(busy(a, b) for a, b in windows) / 1000 / n,
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
        "operators_by_device_time": operator_kernel_times(events, kernels, n),
        "activity_by_step": [
            {
                "step": step["name"],
                "window_ms": step["dur"] / 1000,
                "all_device_ms": activity_union_ms(kernels, step["ts"], step["ts"] + step["dur"]),
                "non_nccl_device_ms": activity_union_ms(
                    [k for k in kernels if "nccl" not in k.get("name", "").lower()],
                    step["ts"],
                    step["ts"] + step["dur"],
                ),
            }
            for step in steps
        ],
        "launch_phase_kernel_ms_per_step": launch_phase_times(events, kernels, n),
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
