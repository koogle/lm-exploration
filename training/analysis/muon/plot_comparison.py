"""Rebuild the matched-token validation plot from retained diagnostic records.

The validation JSONL inputs are archived in the HF dataset
`jfrick/lm-exploration-results` under `pr-24/training/results/`. Download them into
an ignored local folder and pass it explicitly, e.g. from `training/`:

    uv run --with matplotlib python analysis/muon/plot_comparison.py \
      --input-dir artifacts/pr-24/results --output artifacts/pr-24/muon-curve.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--prefix", default="muon-124m-100mt-20261019-b")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = args.input_dir
prefix = args.prefix
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
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180, bbox_inches="tight")
