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
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for cap, r in runs.items():
    steps = [p["step"] for p in r["curve"]]
    label = f"{cap}-token training cap"
    axes[0].plot(
        steps,
        [p["validation"]["accepted_beyond_anchor"] for p in r["curve"]],
        marker="o",
        label=label,
    )
    axes[1].plot(steps, [p["validation"]["nll"] for p in r["curve"]], marker="o", label=label)
axes[0].axhline(
    2, color="grey", linestyle=":", label="Macro criterion; domain floors also required"
)
axes[0].set(
    ylabel="Validation consecutive correct NEW tokens / 7", xlabel="Optimizer updates", ylim=(0, 4)
)
axes[1].set(ylabel="Validation new-token NLL (lower is better)", xlabel="Optimizer updates")
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("IN PROGRESS: fresh 79M drafter for frozen Qwen2.5-0.5B; 49,152 examples")
fig.text(
    0.5,
    0.01,
    "Same random initialization and long validation trajectories. Each fit targets 100,000 updates. Test untouched.",
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
