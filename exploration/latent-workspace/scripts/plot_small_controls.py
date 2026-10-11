"""Plot complete or explicitly partial small-data training/validation diagnostics."""

import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path(sys.argv[1])
with gzip.open(source, "rt") if source.suffix == ".gz" else source.open() as stream:
    report = json.load(stream)
report = report.get("report", report)
arms = report.get("arms", {"baseline (partial)": report})
if len(arms) > 1:
    reference = next(iter(arms.values()))
    for arm in arms.values():
        for key in ("model_revision", "batch_size", "learning_rate", "seed", "layers"):
            assert arm["config"][key] == reference["config"][key], key
        assert arm["data"]["window_hashes"] == reference["data"]["window_hashes"]
        for size in set(arm["curves"]) & set(reference["curves"]):
            assert (
                arm["curves"][size]["train_indices"] == reference["curves"][size]["train_indices"]
            )
sizes = sorted({int(size) for r in arms.values() for size in r["curves"]})
fig, axes = plt.subplots(2, len(sizes), figsize=(max(7, 5 * len(sizes)), 7), squeeze=False)
rows = []
for column, size in enumerate(sizes):
    for label, arm in arms.items():
        curve = arm["curves"].get(str(size))
        if not curve or not curve["checkpoints"]:
            continue
        points = curve["checkpoints"]
        x = [p["step"] * arm["config"]["batch_size"] / size for p in points]
        train = [sum(p["training_sample_nll_by_position"]) / 4 for p in points]
        valid = [p["macro_validation_nll"] for p in points]
        line = axes[0, column].plot(x, train, marker=".", label=label)[0]
        axes[1, column].plot(x, valid, marker=".", color=line.get_color(), label=label)
        last = points[-1]
        rows.append(
            {
                "arm": label,
                "windows": size,
                "step": last["step"],
                "passes": x[-1],
                "train_nll": train[-1],
                "validation_nll": valid[-1],
                "train_accuracy": sum(last["training_sample_accuracy_by_position"]) / 4,
                "best_validation_nll": min(valid),
                "best_step": points[valid.index(min(valid))]["step"],
            }
        )
    axes[0, column].set_title(f"{size:,} distinct training windows")
    for row in range(2):
        ax = axes[row, column]
        ax.set_xscale("log")
        ax.set_xlabel("Sampled windows / distinct windows (log scale)")
        ax.set_ylabel("Training NLL" if row == 0 else "Validation NLL")
        ax.grid(alpha=0.2)
    axes[0, column].legend(fontsize=8)
fig.suptitle("Four-token diffusion drafter: fitting versus generalization")
fig.text(
    0.5,
    0.01,
    "All-masked loss, lower is better. Existing validation reused; no new test evaluation. "
    + (
        "Completed suite."
        if report.get("arms") and report.get("completed")
        else "PARTIAL observations."
    ),
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/small-layer-controls.png")
fig.savefig(destination, dpi=160)
destination.with_suffix(".summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
