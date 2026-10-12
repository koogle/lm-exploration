"""Archive matched live snapshots and plot the response-length comparison."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
    cap: json.loads(Path(f"artifacts/fresh-grid-49152-r{cap}-live.json").read_text())
    for cap in (64, 256)
}
a, b = runs.values()
assert a["initial_drafter_sha256"] == b["initial_drafter_sha256"]
assert a["source_sha256"] == b["source_sha256"]
for split in ("validation", "validation_short"):
    assert a["split_sequence_sha256"][split] == b["split_sequence_sha256"][split]
for key in (
    "model",
    "model_revision",
    "seed",
    "train_limit",
    "batch_size",
    "anchors_per_sequence",
    "learning_rate",
    "schedule_steps",
    "teacher_parent",
    "draft_layers",
    "layers",
    "qk_norm",
):
    assert a["config"][key] == b["config"][key]
assert all(r["warm_start"] is None for r in runs.values())
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for row, (cap, r) in enumerate(runs.items()):
    # Omit initialization to make the subsequent generalization gap readable.
    curve = [p for p in r["curve"] if p["step"] > 0]
    steps = [p["step"] / 1000 for p in curve]
    for col, (metric, ylabel) in enumerate(
        (
            ("nll", "New-token NLL (lower is better)"),
            ("accepted_beyond_anchor", "Consecutive correct NEW tokens / 7"),
        )
    ):
        ax = axes[row, col]
        ax.plot(
            steps,
            [p["validation"][metric] for p in curve],
            marker="o",
            markersize=3,
            label="Validation: 192 responses",
        )
        ax.plot(
            steps,
            [p["training_diagnostic"][metric] for p in curve],
            linestyle="--",
            color="orange",
            label="Training diagnostic: 12 responses",
        )
        chosen = (
            min(curve, key=lambda p: p["validation"][metric])
            if col == 0
            else max(curve, key=lambda p: p["validation"][metric])
        )
        x, y = chosen["step"] / 1000, chosen["validation"][metric]
        ax.scatter([x], [y], color="green", s=60, zorder=5)
        ax.annotate(
            f"Best {y:.2f} at {x:.1f}K",
            (x, y),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
        )
        ax.set(title=f"{cap}-token training cap", ylabel=ylabel, ylim=(0, 7))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[row, 1].axhline(2, color="grey", linestyle=":")
for ax in axes[1]:
    ax.set_xlabel("Optimizer updates (thousands)")
fig.suptitle("IN PROGRESS: fresh 79M drafter for frozen Qwen2.5-0.5B; 49,152 examples")
fig.text(
    0.5,
    0.01,
    "Common long validation; fixed small training diagnostic is not a full-corpus audit. Initialization omitted. Test untouched.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
output = Path("results/fresh-grid-50k-interim")
fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
output.with_suffix(".json").write_text(
    json.dumps(
        {
            "status": "IN_PROGRESS",
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "matched_initial_weights_source_and_validation": True,
            "source_report_sha256": {
                str(cap): hashlib.sha256(
                    Path(f"artifacts/fresh-grid-49152-r{cap}-live.json").read_bytes()
                ).hexdigest()
                for cap in runs
            },
            "reports": runs,
        },
        indent=2,
    )
    + "\n"
)
