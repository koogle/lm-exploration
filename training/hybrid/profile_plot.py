"""Plot an existing trace and all-rank exclusive ledger; no GPU required.

python -m model_hybrid.profile_plot --trace trace.json.gz \
    --headline all-rank-headline.json --output /path/timing

Requires matplotlib. Emits PNG, SVG and the full kernel-duration table as JSON.
Kernel sums overlap; the exclusive ledger is the only wall-time partition.
"""

import argparse
import collections
import gzip
import json
from pathlib import Path

from .profile_categories import kernel_category

LABELS = {
    "matmul": "Dense matrix kernels (incl. output heads + optimizer)",
    "kda_fla": "KDA recurrence + gates + L2 normalization",
    "compiled_elementwise": "Compiled elementwise / reductions / casts",
    "other": "Other native kernels (details below)",
    "communication": "NCCL collectives (can include waiting)",
    "kda_convolution": "KDA convolution forward + backward",
    "loss_softmax_gradient": "Loss softmax gradient",
    "flex_attention": "Full attention + fused surrounding operations",
    "memory": "Copies / fills / memory operations",
    "optimizer": "Explicit optimizer / multi-tensor kernels only",
}


def kernel_durations(trace):
    events = trace["traceEvents"]
    steps = [
        e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep")
    ]
    if not steps:
        raise ValueError("Trace has no complete profiler step windows")
    times = collections.Counter()
    calls = collections.Counter()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        for step in steps:
            duration = max(
                0,
                min(event["ts"] + event["dur"], step["ts"] + step["dur"])
                - max(event["ts"], step["ts"]),
            )
            if duration:
                times[event["name"]] += duration / 1000 / len(steps)
                calls[event["name"]] += 1 / len(steps)
    return [
        {
            "name": name,
            "ms_per_step": ms,
            "calls_per_step": calls[name],
            "category": kernel_category(name),
        }
        for name, ms in times.most_common()
    ]


def render(trace_path, headline_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with gzip.open(trace_path) as handle:
        kernels = kernel_durations(json.load(handle))
    headline = json.loads(Path(headline_path).read_text())
    categories = collections.Counter()
    for row in kernels:
        categories[row["category"]] += row["ms_per_step"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "trace": str(trace_path),
                "headline": str(headline_path),
                "scope": "Per-step means. Kernel duration sums overlap across streams; exclusive ledger counts each instant once.",
                "categories_ms": dict(categories),
                "kernels": kernels,
                "exclusive": headline,
            },
            indent=2,
        )
        + "\n"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(18, 22), facecolor="#f8fafc")
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1, 1.35, 1.7],
        left=0.33,
        right=0.93,
        top=0.91,
        bottom=0.06,
        hspace=0.40,
    )
    fig.text(0.04, 0.972, "Where the eight H100s spend a training step", fontsize=25, weight="bold")
    fig.text(
        0.04,
        0.946,
        "960M KDA • real 50/40/10 data • context 4,096 • global batch 256 • microbatch 6\n"
        "Capture: kda-allrank-profile-h100x8-20261106-02 • mean of three captured steps",
        fontsize=12,
    )
    ax = fig.add_subplot(grid[0])
    keys = ["compute_only", "communication_only", "memops_only", "overlap", "idle"]
    labels = [
        "Non-NCCL kernels only",
        "NCCL only",
        "Memory ops only",
        "Overlapping activity",
        "No recorded GPU activity",
    ]
    colors = ["#2563eb", "#a855f7", "#06b6d4", "#10b981", "#cbd5e1"]
    for i, row in enumerate(headline["ranks"]):
        values = row["mean_exclusive_ms"]
        assert abs(sum(values.values()) - row["mean_window_ms"]) < 1e-6
        left = 0
        for key, label, color in zip(keys, labels, colors, strict=True):
            value = values[key]
            ax.barh(i, value, left=left, color=color, label=label if i == 0 else None)
            left += value
        ax.text(left + 20, i, f"{values['idle']:.0f} ms idle", va="center", fontsize=10)
    ax.set_yticks(range(len(headline["ranks"])), [f"GPU {r['rank']}" for r in headline["ranks"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 3400)
    ax.set_xlabel(
        "Milliseconds per step — exclusive allocation; segments regrouped, not chronological"
    )
    ax.set_title(
        "1. Every millisecond counted once: ~3,037 ms per GPU", loc="left", pad=14, weight="bold"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.20), ncol=3, frameon=False, fontsize=10)

    ax = fig.add_subplot(grid[1])
    items = categories.most_common()
    ax.barh(range(len(items)), [v for _, v in items], color="#2563eb")
    ax.set_yticks(range(len(items)), [LABELS[k] for k, _ in items], fontsize=11)
    for i, (_, value) in enumerate(items):
        ax.text(value + 12, i, f"{value:,.1f} ms", va="center", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(categories.values()) * 1.18)
    ax.set_xlabel("Summed kernel duration on GPU 0 (ms/step); overlapping streams count separately")
    ax.set_title("2. What runs while the GPU is busy", loc="left", pad=14, weight="bold")

    ax = fig.add_subplot(grid[2])
    # Individual dense kernels are implementation variants, not functional layers.
    # Show the largest non-GEMM names so the remaining work is inspectable.
    detail = [r for r in kernels if r["category"] not in ("matmul", "communication")][:24]
    names = []
    for row in detail:
        name = row["name"]
        if "CUDAFunctor_add<float>" in name:
            name = "Native FP32 elementwise addition"
        elif "softmax_gradient_kernel" in name:
            name = "Loss: softmax_gradient_kernel"
        elif "kda_conv_backward" in name:
            name = "KDA: convolution backward"
        elif "kda_conv_forward" in name:
            name = "KDA: convolution forward"
        elif "CUDAFunctor_add<c10::BFloat16>" in name:
            name = "Native BF16 elementwise addition"
        name = name.replace("triton_", "").replace("_kernel", "")
        names.append(name if len(name) <= 66 else name[:63] + "…")
    ax.barh(range(len(detail)), [r["ms_per_step"] for r in detail], color="#0d9488")
    ax.set_yticks(range(len(detail)), names, fontsize=9)
    for i, row in enumerate(detail):
        ax.text(row["ms_per_step"] + 1, i, f"{row['ms_per_step']:.1f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["ms_per_step"] for r in detail) * 1.12)
    ax.set_xlabel("Summed kernel duration on GPU 0 (ms/step); rows are subsets of panel 2")
    ax.set_title("3. Largest non-GEMM, non-NCCL kernels", loc="left", pad=14, weight="bold")
    for ax in fig.axes:
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.text(
        0.04,
        0.019,
        "Profiler overhead: ~3.1%. These are steady steps, excluding startup, evaluation and checkpoints.\n"
        "GPU activity is not tensor-core efficiency. Name-based categories are operation groups, not proof of a bandwidth or compute bottleneck.\n"
        "Optimizer work also appears under GEMMs and elementwise; KDA projections also appear under GEMMs. Full kernel names and counts: companion JSON.",
        fontsize=10,
        color="#475569",
    )
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--headline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.trace, args.headline, args.output)


if __name__ == "__main__":
    main()
