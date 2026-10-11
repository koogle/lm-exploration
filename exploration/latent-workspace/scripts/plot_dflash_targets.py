"""Compare recorded drafting quality; different tokenizers are not a paired task."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline")
parser.add_argument("larger")
parser.add_argument("--output", default="results/dflash-qwen35-2b-comparison.png")
args = parser.parse_args()
if Path(args.output).with_suffix(".json").resolve() in {
    Path(p).resolve() for p in (args.baseline, args.larger)
}:
    raise ValueError("analysis output must not overwrite an input report")
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
            "training_completed": report.get("training_completed", False),
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
fig.suptitle(
    ("IN PROGRESS — " if not b.get("training_completed") else "")
    + "Qwen target comparison\nTokenizer, prompt format, architecture and draft capacity also change"
)
fig.tight_layout()
fig.savefig(args.output, dpi=160, bbox_inches="tight")
summary["input_sha256"] = {
    p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.baseline, args.larger)
}
if all("selected_validation" in r for r in (a, b)):
    examples = [{e["group"]: e for e in r["selected_validation"]["examples"]} for r in (a, b)]
    groups = sorted(set(examples[0]) & set(examples[1]))
    paired = {
        "shared_prompts": len(groups),
        "note": "Validation-selected checkpoints; same source groups but different target responses/tokenizers/prompts. No independent test or pure size effect.",
    }
    for report, lookup in zip((a, b), examples):
        by_domain = {}
        for domain in ("prose", "code", "translation"):
            rows = [lookup[g] for g in groups if lookup[g]["domain"] == domain]
            blocks = sum(e["blocks"] for e in rows)
            by_domain[domain] = {
                "prompts": len(rows),
                "blocks": blocks,
                "accepted_new": sum(e["blocks"] * e["accepted_beyond_anchor"] for e in rows)
                / blocks,
            }
        paired[report["config"]["model"]] = {
            "best_step": report["best_step"],
            "by_domain": by_domain,
            "accepted_new": sum(d["accepted_new"] for d in by_domain.values()) / 3,
        }
    summary["selected_shared_prompt_comparison"] = paired
Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
