"""Compare fixed-corpus fidelity studies; partial curves remain explicitly labelled."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path)
parser.add_argument("--output", type=Path, default=Path("results/dflash-fidelity-curves.png"))
args = parser.parse_args()
runs = [json.loads(p.read_text()) for p in args.reports]
for run in runs[1:]:
    for key in (
        "model",
        "model_revision",
        "block_size",
        "gamma",
        "batch_size",
        "anchors_per_sequence",
        "learning_rate",
        "seed",
    ):
        if run["config"][key] != runs[0]["config"][key]:
            raise ValueError(f"Unmatched comparison: {key}")
    if run["split_sequence_sha256"] != runs[0]["split_sequence_sha256"]:
        raise ValueError("Comparison requires identical train and validation tokens")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for run in runs:
    cfg = run["config"]
    label = f"{cfg['draft_layers']} draft layers, {'QK norm' if cfg.get('qk_norm') else 'no QK norm'}, {len(cfg['layers'])} feature layers"
    steps = [r["step"] for r in run["curve"]]
    (line,) = axes[0].plot(steps, [r["validation"]["nll"] for r in run["curve"]], label=label)
    axes[0].plot(
        steps,
        [r["training_diagnostic"]["nll"] for r in run["curve"]],
        "--",
        color=line.get_color(),
        alpha=0.6,
    )
    axes[1].plot(
        steps, [r["validation"]["accepted_beyond_anchor"] for r in run["curve"]], label=label
    )
    selected = next(r["validation"] for r in run["curve"] if r["step"] == run["best_step"])
    accuracy = np.mean(
        [r["accuracy_by_draft_position"] for r in selected["by_domain"].values()], axis=0
    )
    axes[2].plot(range(1, cfg["block_size"]), accuracy, marker="o", label=label)
axes[0].set(xlabel="Updates", ylabel="New-token NLL (lower is better)")
axes[1].set(xlabel="Updates", ylabel="Consecutive correct NEW tokens / 7", ylim=(0, 7))
axes[1].axhline(2, color="gray", linestyle=":", label="Working macro quality gate")
axes[2].set(xlabel="NEW position", ylabel="Selected validation token accuracy", ylim=(0, 1))
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
partial = any(not r.get("training_completed") for r in runs)
fig.suptitle("Our drafter for frozen Qwen2.5-0.5B" + (" — IN PROGRESS" if partial else ""))
fig.text(
    0.5,
    0.01,
    "Identical teacher sequences. Dashed NLL: 12 training diagnostics only. Validation selected; test remains sealed.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(args.output, dpi=160)
print(args.output)
