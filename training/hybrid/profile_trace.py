"""Audit a benchmark's compressed Chrome trace without allocating a GPU.

    python -m model_hybrid.profile_trace --trace /path/trace.json.gz --output /path/audit.json

Host ranges are clipped to profiled steps. Device activity is a union, avoiding
counting overlapping streams twice. These instrumented times are diagnostic;
use unprofiled same-allocation runs for speed comparisons.
"""

import argparse
import base64
import bisect
import collections
import copy
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
        "scope": "Profiler-instrumented supplied trace. Host phases overlap; GPU overlap is activity during each host phase, not exclusive kernel attribution. Command-buffer overhead can be induced by profiling.",
    }
    return r


def exclusive_ledger(events, lo, hi):
    """Partition a device window; concurrency within a class counts only once."""
    if hi <= lo:
        raise ValueError("The step window must have positive duration")
    changes = collections.defaultdict(collections.Counter)
    changes[lo]
    changes[hi]
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        a, b = max(lo, event["ts"]), min(hi, event["ts"] + event["dur"])
        if b <= a:
            continue
        kind = (
            ("communication" if "nccl" in event.get("name", "").lower() else "compute")
            if category == "kernel"
            else "memops"
        )
        changes[a][kind] += 1
        changes[b][kind] -= 1
    totals = dict.fromkeys(
        ("compute_only", "communication_only", "memops_only", "overlap", "idle"), 0.0
    )
    segments, active, previous = [], collections.Counter(), lo
    for point in sorted(changes):
        if point > previous:
            kinds = sorted(k for k, count in active.items() if count > 0)
            label = "overlap" if len(kinds) > 1 else kinds[0] + "_only" if kinds else "idle"
            totals[label] += (point - previous) / 1000
            if segments and segments[-1][2:] == [label, kinds]:
                segments[-1][1] = (point - lo) / 1000
            else:
                segments.append([(previous - lo) / 1000, (point - lo) / 1000, label, kinds])
        active.update(changes[point])
        previous = point
    window = (hi - lo) / 1000
    return {
        "window_ms": window,
        "exclusive_ms": totals,
        "residual_ms": window - sum(totals.values()),
        "segments": segments,
    }


def host_details(events, lo, hi, limit=50000):
    """Bound displayed host spans; totals retain all complete-duration host events."""
    selected, totals = [], collections.Counter()
    launches = []
    for e in events:
        cat = e.get("cat")
        if e.get("ph") != "X" or cat not in (
            "cpu_op",
            "cuda_runtime",
            "cuda_driver",
            "python_function",
        ):
            continue
        a, b = max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])
        if b <= a:
            continue
        totals[(cat, e["name"])] += (b - a) / 1000
        args = e.get("args", {})
        stack = args.get(
            "Call stack", args.get("callstack", args.get("stack", "No recorded stack"))
        )
        detail = json.dumps(
            {
                "correlation": args.get("correlation"),
                "external_id": args.get("External id"),
                "stack": stack,
            }
        )
        span = [
            (a - lo) / 1000,
            (b - lo) / 1000,
            f"Host {cat} thread {e.get('tid', '?')}",
            e["name"],
            detail,
        ]
        selected.append(span)
        if (
            cat in ("cuda_runtime", "cuda_driver")
            and "correlation" in args
            and any(token in e["name"].lower() for token in ("launch", "memcpy", "memset"))
        ):
            launches.append(span)
    kept = sorted(
        sorted(selected, key=lambda x: x[1] - x[0], reverse=True)[:limit], key=lambda x: x[0]
    )
    return {
        "spans": kept,
        "omitted_spans": max(0, len(selected) - limit),
        "totals": [[cat, name, ms] for (cat, name), ms in totals.most_common()],
        "launches": sorted(launches, key=lambda x: x[0]),
    }


def timeline_data(trace, rank):
    events = trace["traceEvents"]
    steps = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("ProfilerStep")
        ),
        key=lambda e: e["ts"],
    )
    if not steps:
        raise ValueError("Trace has no complete ProfilerStep windows")
    operators = {
        e["args"]["External id"]: e.get("name", "unattributed")
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    result = []
    for step in steps:
        lo, hi = step["ts"], step["ts"] + step["dur"]
        ledger = exclusive_ledger(events, lo, hi)
        host_info = host_details(events, lo, hi)
        times = [x[0] for x in host_info["launches"]]
        gaps = []
        for segment in ledger["segments"]:
            if segment[2] == "idle":
                before = bisect.bisect_right(times, segment[0]) - 1
                after = bisect.bisect_left(times, segment[1])
                within = host_info["launches"][before + 1 : after]
                gaps.append(
                    {
                        "start_ms": segment[0],
                        "end_ms": segment[1],
                        "submissions_inside_gap": len(within),
                        "first_submission_inside_gap": within[0] if within else None,
                        "last_submission_inside_gap": within[-1] if within else None,
                        "preceding_launch": host_info["launches"][before] if before >= 0 else None,
                        "following_launch": host_info["launches"][after]
                        if after < len(times)
                        else None,
                    }
                )
        spans, grouped = [], collections.Counter()
        for e in events:
            cat = e.get("cat")
            device = cat in ("kernel", "gpu_memcpy", "gpu_memset")
            host = cat == "user_annotation" and e.get("name", "").startswith("train/")
            if e.get("ph") != "X" or not (device or host):
                continue
            a, b = max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])
            if b <= a:
                continue
            op = (
                operators.get(e.get("args", {}).get("External id"), "unattributed")
                if device
                else "host annotation"
            )
            lane = (
                "GPU stream " + str(e.get("args", {}).get("stream", e.get("tid", "unknown")))
                if device
                else "CPU " + e["name"]
            )
            correlation = e.get("args", {}).get("correlation")
            launch = launches.get(correlation)
            detail = {
                "correlation": correlation,
                "launch": launch.get("name") if launch else None,
                "launch_start_ms": (launch["ts"] - lo) / 1000 if launch else None,
                "launch_duration_ms": launch.get("dur", 0) / 1000 if launch else None,
                "launch_thread": launch.get("tid") if launch else None,
            }
            spans.append(
                [(a - lo) / 1000, (b - lo) / 1000, lane, e["name"], op, json.dumps(detail)]
            )
            if device:
                grouped[(op, e["name"])] += (b - a) / 1000
        ledger.update(
            {
                "name": step["name"],
                "spans": spans,
                "host_spans": host_info["spans"],
                "host_omitted_spans": host_info["omitted_spans"],
                "host_totals": host_info["totals"],
                "idle_gap_context": gaps,
                "kernel_totals": [[op, name, ms] for (op, name), ms in grouped.most_common()],
                "unattributed_kernel_sum_ms": sum(
                    ms for (op, _), ms in grouped.items() if op == "unattributed"
                ),
            }
        )
        result.append(ledger)
    return {
        "rank": rank,
        "steps": result,
        "clock_metadata": {
            k: trace.get(k) for k in ("baseTimeNanoseconds", "host_name", "distributedInfo")
        },
    }


def render_timeline(ranks, output):
    compact = copy.deepcopy(ranks)
    strings, indices = [], {}
    for rank in compact:
        for step in rank.get("steps", []):
            for field in ("spans", "host_spans"):
                for span in step.get(field, []):
                    for i in range(2, len(span)):
                        value = span[i]
                        if value not in indices:
                            indices[value] = len(strings)
                            strings.append(value)
                        span[i] = indices[value]
    # Intern repeated names for browser load cost; escape script text, never HTML.
    payload = (
        json.dumps({"ranks": compact, "strings": strings})
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("profile_timeline.html").read_text()
    Path(output).write_text(
        template.replace(
            "__TRACE_DATA__", base64.b64encode(gzip.compress(payload.encode())).decode()
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", required=True, action="append", help="Repeat for independent per-rank traces"
    )
    parser.add_argument(
        "--rank", action="append", help="Rank labels in trace order; defaults to input indices"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--html")
    args = parser.parse_args()
    labels = args.rank or [str(i) for i in range(len(args.trace))]
    if len(labels) != len(args.trace) or len(set(labels)) != len(labels):
        parser.error("Supply one distinct rank label per trace")
    ranks, audits = [], []
    for path, rank in zip(args.trace, labels, strict=True):
        with gzip.open(path) as handle:
            trace = json.load(handle)
        ranks.append(timeline_data(trace, rank))
        audits.append({"rank": rank, "audit": summarize(trace)})
    result = {
        "ranks": ranks,
        "audits": audits,
        "scope": "Profiler-instrumented traces; each rank has its own clock/window. No cross-rank alignment or causal attribution of CPU waits. Missing ranks are not inferred. Kernel sums overlap and are separate from the exclusive ledger.",
    }
    Path(args.output).write_text(json.dumps(result) + "\n")
    if args.html:
        render_timeline(ranks, args.html)


if __name__ == "__main__":
    main()
