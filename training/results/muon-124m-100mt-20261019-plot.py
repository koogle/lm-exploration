"""Rebuild the matched-token validation plot from retained diagnostic records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).parent
prefix = "muon-124m-100mt-20261019-b"
fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
for name, label, color in (("adamw", "AdamW", "#3454d1"), ("muon", "Muon + AdamW", "#cf5c36")):
    rows = []
    for segment in (name + "-start", name):
        rows.extend(
            json.loads(line)
            for line in (folder / f"{prefix}-{segment}-validation.jsonl").read_text().splitlines()
        )
    by_step = {r["step"]: r for r in rows}
    points = sorted(by_step.values(), key=lambda r: r["step"])
    ax.plot(
        [r["tokens"] / 1e6 for r in points],
        [r["weighted_loss"] for r in points],
        marker="o",
        markersize=3,
        label=label,
        color=color,
    )
ax.set(
    xlabel="Nonpadding training tokens (millions)",
    ylabel="Weighted diagnostic loss (nats)",
    title="Fresh 124.4M model: matched-token optimizer comparison",
)
ax.grid(alpha=0.2)
ax.legend()
fig.text(
    0.5,
    -0.035,
    "One seed · one fixed validation window per source · eight H100s",
    ha="center",
    fontsize=9,
)
fig.savefig(folder / "muon-124m-100mt-20261019-curve.png", dpi=180, bbox_inches="tight")
