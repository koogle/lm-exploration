"""Compare recorded drafting quality; different tokenizers are not a paired task."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline")
parser.add_argument("larger")
parser.add_argument("--output", default="results/dflash-qwen35-2b-comparison.png")
args = parser.parse_args()
a, b = [json.loads(Path(p).read_text()) for p in (args.baseline, args.larger)]
common = sorted({x["step"] for x in a["curve"]} & {x["step"] for x in b["curve"]})
if not common:
    raise ValueError("no shared measured updates")
step = common[-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
summary = {
    "comparison_step": step,
    "note": "Different targets, tokenizers and draft widths; not size-only or paired-label evidence.",
    "models": [],
}
for report, color in zip((a, b), ("#3573b9", "#df7b24")):
    name = report["config"]["model"].split("/")[-1]
    rows = [r for r in report["curve"] if r["step"] <= step]
    for key, style, label in [
        ("validation", "-", "validation"),
        ("training_diagnostic", "--", "train sample"),
    ]:
        axes[0].plot(
            [r["step"] for r in rows],
            [r[key]["accepted_beyond_anchor"] for r in rows],
            style,
            color=color,
            label=f"{name} {label}",
        )
    row = next(r for r in rows if r["step"] == step)
    domains = row["validation"]["by_domain"]
    accuracy = [
        sum(d["accuracy_by_draft_position"][i] for d in domains.values()) / len(domains)
        for i in range(7)
    ]
    axes[1].plot(range(1, 8), accuracy, "o-", color=color, label=name)
    summary["models"].append(
        {
            "model": name,
            "trainable_parameters": report["trainable_parameters"],
            "validation": row["validation"],
            "training_diagnostic": row["training_diagnostic"],
            "training_completed": report["training_completed"],
        }
    )
axes[0].set(
    xlabel="Optimizer updates",
    ylabel="Mean consecutive NEW matches / 7",
    title="Train sample and held-out drafting",
)
axes[0].axhline(1, color="gray", linestyle=":", linewidth=1)
axes[1].set(
    xlabel="NEW token position after anchor",
    ylabel="Held-out token accuracy",
    title=f"Same update count: {step:,}",
    ylim=(0, 1),
)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("Qwen target comparison — tokenizer, architecture and draft capacity also change")
fig.tight_layout()
fig.savefig(args.output, dpi=160, bbox_inches="tight")
Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
