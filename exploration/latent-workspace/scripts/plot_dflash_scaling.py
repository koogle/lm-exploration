"""Validate matched runs, plot curves, and bootstrap paired validation prompts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs=2, type=Path)
parser.add_argument("--output", type=Path, default=Path("results/dflash-data-scaling"))
parser.add_argument(
    "--partial", action="store_true", help="Plot incomplete curves; no uncertainty estimates"
)
args = parser.parse_args()
runs = sorted(
    (json.loads(p.read_text()) for p in args.reports), key=lambda r: r["config"]["train_limit"]
)
for key in (
    "model",
    "model_revision",
    "layers",
    "draft_layers",
    "block_size",
    "gamma",
    "batch_size",
    "anchors_per_sequence",
    "learning_rate",
    "warmup_ratio",
    "seed",
    "teacher_parent",
    "eval_every",
):
    assert runs[0]["config"][key] == runs[1]["config"][key], f"unmatched setting: {key}"
for key, default in (
    ("qk_norm", False),
    ("warm_start_from", ""),
    ("retain_training_parent", ""),
    ("schedule_steps", 0),
):
    assert runs[0]["config"].get(key, default) == runs[1]["config"].get(key, default), key
assert runs[0].get("training_goal_step", runs[0]["config"]["steps"]) == runs[1].get(
    "training_goal_step", runs[1]["config"]["steps"]
), "unmatched total update budget"
assert runs[0]["initial_drafter_sha256"] == runs[1]["initial_drafter_sha256"]
assert (
    runs[0]["split_sequence_sha256"]["validation"] == runs[1]["split_sequence_sha256"]["validation"]
)
if not args.partial:
    assert all(r["training_completed"] for r in runs)
assert runs[0]["source_sha256"] == runs[1]["source_sha256"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for color, r in zip(("tab:blue", "tab:orange"), runs):
    label = f"{r['config']['train_limit']:,} responses"
    steps = [p["step"] for p in r["curve"]]
    for metric, style in (("training_diagnostic", "--"), ("validation", "-")):
        axes[0].plot(
            steps,
            [p[metric]["nll"] for p in r["curve"]],
            color=color,
            linestyle=style,
            label=f"{label}: {'train sample' if style == '--' else 'validation'}",
        )
    axes[1].plot(
        steps,
        [p["validation"]["accepted_beyond_anchor"] for p in r["curve"]],
        color=color,
        label=label,
    )
    best = r.get("selected_validation") or next(
        p["validation"] for p in r["curve"] if p["step"] == r["best_step"]
    )
    accuracy = np.mean(
        [v["accuracy_by_draft_position"] for v in best["by_domain"].values()], axis=0
    )
    axes[2].plot(
        range(1, 8), accuracy, marker="o", color=color, label=f"{label} (step {r['best_step']:,})"
    )
axes[0].set(xlabel="Optimizer updates", ylabel="New-token NLL (lower is better)")
axes[1].set(
    xlabel="Optimizer updates",
    ylabel="Validation accepted new tokens per block (max 7)",
    ylim=(
        0,
        max(
            1.05,
            1.1 * max(p["validation"]["accepted_beyond_anchor"] for r in runs for p in r["curve"]),
        ),
    ),
)
axes[1].axhline(1, color="grey", linestyle=":", linewidth=1, label="1 new token (reference)")
axes[2].set(
    xlabel="New draft position; known anchor excluded",
    ylabel="Validation token accuracy",
    ylim=(0, 1),
)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
fig.suptitle(
    "DFlash-style Qwen2.5-0.5B: different response-corpus sizes"
    + (" — IN PROGRESS" if args.partial else "")
)
fig.text(
    0.5,
    0.01,
    "Same initialization and validation cache. Training curve: 12 diagnostic prompts. Validation-selected checkpoints; not test results.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(args.output.with_suffix(".png"), dpi=160, bbox_inches="tight", pad_inches=0.2)
if args.partial:
    print("Interim curves only; jobs are still running. " + str(args.output.with_suffix(".png")))
    raise SystemExit(0)

# Resample source groups within each domain, preserving pairing and block weights.
examples = [{e["group"]: e for e in r["selected_validation"]["examples"]} for r in runs]
assert examples[0].keys() == examples[1].keys()
rng = np.random.default_rng(20261011)
differences = np.zeros((10000, 2))
point = np.zeros(2)
for domain in ("prose", "code", "translation"):
    groups = sorted(g for g, e in examples[0].items() if e["domain"] == domain)
    weights = np.array([examples[0][g]["blocks"] for g in groups])
    assert all(examples[1][g]["blocks"] == examples[0][g]["blocks"] for g in groups)
    delta = np.array(
        [
            [examples[1][g][key] - examples[0][g][key] for key in ("accepted_beyond_anchor", "nll")]
            for g in groups
        ]
    )
    indices = rng.integers(len(groups), size=(10000, len(groups)))
    sampled_weights = weights[indices]
    differences += (
        (delta[indices] * sampled_weights[:, :, None]).sum(1) / sampled_weights.sum(1)[:, None] / 3
    )
    point += (delta * weights[:, None]).sum(0) / weights.sum() / 3
expected_difference = [
    runs[1]["selected_validation"][key] - runs[0]["selected_validation"][key]
    for key in ("accepted_beyond_anchor", "nll")
]
np.testing.assert_allclose(point, expected_difference, atol=1e-5, rtol=1e-5)
summary = {
    "matched_initialization_and_validation": True,
    "source_sha256": runs[0]["source_sha256"],
    "validation_sequence_sha256": runs[0]["split_sequence_sha256"]["validation"],
    "reports": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports},
    "paired_validation_difference_large_minus_small": {
        key: {
            "estimate": float(point[i]),
            "bootstrap_95_percent": np.quantile(differences[:, i], [0.025, 0.975]).tolist(),
        }
        for i, key in enumerate(("accepted_new_tokens", "nll"))
    },
    "uncertainty_note": "10000 paired source-group bootstrap samples, stratified by domain, block-weighted within domains. Conditional on validation-selected checkpoints; does not capture model-selection bias or seed variation. Not a test-set result.",
}
args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
