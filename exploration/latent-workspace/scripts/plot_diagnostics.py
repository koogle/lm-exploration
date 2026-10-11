"""Plot the controlled 96-window memorization check (training metrics only)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for run, label in (
    ("data-memorize-01", "Original shared projection"),
    ("data-position-memorize-01", "Position-specific projections"),
):
    report = json.loads(Path(f"results/{run}.json").read_text())["report"]
    rows = report["curves"]["96"]["checkpoints"]
    steps = [r["step"] for r in rows]
    axes[0].plot(
        steps,
        [sum(r["training_sample_nll_by_position"]) / 4 for r in rows],
        marker="o",
        label=label,
    )
    axes[1].plot(
        steps,
        [100 * sum(r["training_sample_accuracy_by_position"]) / 4 for r in rows],
        marker="o",
        label=label,
    )
axes[0].set(title="Training loss", ylabel="All-masked token NLL (log scale)", yscale="log")
axes[1].set(title="Training token accuracy", ylabel="Percent correct", ylim=(0, 105))
for ax in axes:
    ax.set_xlabel("Optimizer updates")
    ax.grid(alpha=0.2)
axes[1].legend(loc="lower right", fontsize=9)
fig.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.98, wspace=0.24)
fig.suptitle(
    "96-block memorization sanity check · identical data and seed · not held-out quality",
    fontsize=13,
)
fig.savefig("results/decoder-diagnostics.png", dpi=160)
fig.savefig("results/decoder-diagnostics.svg")
