"""Plot NEW-token drafting metrics; never count the supplied anchor as a prediction."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1])
r = json.loads(path.read_text())
points = r["curve"]
steps = [p["step"] for p in points]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for key, label in (
    ("training_diagnostic", "Training sample (12 prompts)"),
    ("validation", "Validation"),
):
    axes[0].plot(steps, [p[key]["nll"] for p in points], marker="o", label=label)
    axes[1].plot(
        steps,
        [p[key]["accepted_beyond_anchor"] for p in points],
        marker="o",
        label=label,
    )
axes[0].set_ylabel("Unweighted new-token NLL (lower is better)")
axes[0].set_xlabel("Optimizer updates")
axes[0].legend(fontsize=8)
axes[1].axhline(1, color="grey", linestyle="--", linewidth=1)
axes[1].set_ylabel("Consecutive NEW tokens accepted / 7")
axes[1].set_xlabel("Optimizer updates")
axes[1].set_ylim(0, 7)
axes[1].legend(fontsize=8)
best = next(p for p in points if p["step"] == r["best_step"])
for domain, v in best["validation"]["by_domain"].items():
    axes[2].plot(range(1, 8), v["accuracy_by_draft_position"], marker="o", label=domain)
axes[2].set_xlabel("New draft position (anchor excluded)")
axes[2].set_ylabel("Held-out teacher-token accuracy")
axes[2].set_title(f"Validation-selected checkpoint: {r['best_step']:,} updates", fontsize=9)
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle("DFlash-style Qwen2.5-0.5B: target-generated training responses")
fig.text(
    0.5,
    0.01,
    "Teacher-trajectory metrics; actual verified rollout is reported separately. Small pilot, not a speedup qualification.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(path.with_suffix(".png"), dpi=160)
print(path.with_suffix(".png"))
