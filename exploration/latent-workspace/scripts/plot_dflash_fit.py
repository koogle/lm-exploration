"""Plot selected and final full-corpus fitting audits, excluding supplied anchors."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("summary", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
rows = summary["results"]
sizes = sorted(rows, key=int)
x = np.arange(len(sizes))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, checkpoint, title in zip(axes, ("best", "latest"), ("Validation-selected", "Final update")):
    for offset, split, color in (
        (-0.18, "training", "tab:blue"),
        (0.18, "validation", "tab:orange"),
    ):
        values = [rows[size][checkpoint][split]["accepted_beyond_anchor"] for size in sizes]
        bars = ax.bar(x + offset, values, width=0.36, color=color, label=split.capitalize())
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(2, color="grey", linestyle=":", label="Validation quality gate")
    ax.set(
        title=title,
        xticks=x,
        xticklabels=[f"{int(size):,} responses" for size in sizes],
        ylim=(0, 7.5),
        ylabel="Consecutive correct NEW tokens per block (max 7)",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
fig.suptitle(
    f"Our {summary['trainable_parameters'] / 1e6:.0f}M drafter for frozen "
    f"{summary['model'].split('/')[-1]}: full-corpus fitting audit"
)
fig.text(
    0.5,
    0.01,
    "Every train/validation response, up to eight fixed anchors each. Known anchor excluded; test sealed.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(args.output, dpi=160, bbox_inches="tight", pad_inches=0.2)
print(args.output)
