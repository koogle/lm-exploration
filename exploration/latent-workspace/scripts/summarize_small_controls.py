"""Summarize full fitting and recorded validation optima; no test access."""

import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with gzip.open(root / "results/small-layer-controls-01.json.gz", "rt") as stream:
    suite = json.load(stream)["report"]
reference = suite["arms"]["baseline"]
for name, arm in suite["arms"].items():
    assert (
        arm["source_sha256"] == "06d951e2d9178a6d7ffe72f95c659579293426e222ee934bddf57aebe0e961e8"
    )
    assert arm["data"]["window_hashes"] == reference["data"]["window_hashes"]
    for size, curve in arm["curves"].items():
        other = reference["curves"][size]
        assert curve["train_indices"] == other["train_indices"]
        assert [(p["step"], p["masked_supervised_tokens_seen"]) for p in curve["checkpoints"]] == [
            (p["step"], p["masked_supervised_tokens_seen"]) for p in other["checkpoints"]
        ]
        assert all(p["training_diagnostic_windows"] == int(size) for p in curve["checkpoints"])
audit = {
    "completed": suite["completed"],
    "arms": list(suite["arms"]),
    "source_sha256": reference["source_sha256"],
    "launch_commit": "255a6b5",
    "identical_ordered_subsets_and_split_hashes": True,
    "identical_supervised_token_counts_by_checkpoint": True,
    "all_training_windows_measured": True,
    "new_test_evaluation": False,
}
(root / "results/small-layer-controls-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
lines = [
    "# Small-data layer-control results",
    "",
    "Launch source: commit `255a6b5`; report source hashes verified against that tree.",
    "Modal job: [small-layer-controls-01](https://modal.com/apps/radiantai/main/ap-UP6RCX2phjsurjzbFIYumb).",
    "",
    f"Suite completed: **{suite['completed']}**. Frozen Qwen2.5-0.5B, seed 42, all 24 residual layers,",
    "four-token diffusion blocks, batch 32, learning rate 0.0003. Three nested balanced",
    "training subsets, all training rows measured. Existing 192-window validation reused;",
    "no new test evaluation. One seed and a small adaptive validation slice: exploratory only.",
    "",
    "| Windows | Control | Final update | Final train NLL | Train token accuracy | Best recorded validation NLL | At update |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
]
for size in suite["config"]["study_sizes"]:
    for name, arm in suite["arms"].items():
        points = arm["curves"].get(str(size), {}).get("checkpoints", [])
        if not points:
            continue
        last = points[-1]
        best = min(points, key=lambda p: p["macro_validation_nll"])
        loss = sum(last["training_sample_nll_by_position"]) / 4
        accuracy = sum(last["training_sample_accuracy_by_position"]) / 4
        lines.append(
            f"| {size} | {name} | {last['step']} | {loss:.3g} | {accuracy:.1%} | "
            f"{best['macro_validation_nll']:.4f} | {best['step']} |"
        )
lines += [
    "",
    "## Largest subset: domains at each control's selected checkpoint",
    "",
    "| Control | Prose NLL | Code NLL | Translation NLL | Qwen consecutive agreement / 4 |",
    "| --- | ---: | ---: | ---: | ---: |",
]
size = str(max(suite["config"]["study_sizes"]))
for name, arm in suite["arms"].items():
    points = arm["curves"].get(size, {}).get("checkpoints", [])
    if not points:
        continue
    best = min(points, key=lambda p: p["macro_validation_nll"])
    domains = best["validation_by_domain"]
    nll = [domains[d]["nll"] for d in ("prose", "code", "translation")]
    agreement = sum(d["qwen_greedy_prefix_agreement"] for d in domains.values()) / 3
    lines.append(f"| {name} | {nll[0]:.4f} | {nll[1]:.4f} | {nll[2]:.4f} | {agreement:.3f} |")
lines += [
    "",
    "![Training and validation curves](small-layer-controls.png)",
    "",
    "The x-axis is sampled windows divided by distinct windows; sampling uses replacement.",
    "Reporting starts at update 25, so best recorded loss is not a guarantee that an earlier",
    "optimum was captured. Training NLL and accuracy use fully masked four-token prediction.",
    "The final-state readout shortcut stays present even when workspace layers are dropped.",
    "Dropout is off at evaluation; zero-content attention remains enabled in null controls.",
    "Matching initialization, sampling, target corruption, source windows and update budgets",
    "is a control, not a replacement for replication. Low train loss demonstrates fitting,",
    "not generalizable future information or useful generation speed.",
    "",
    "[Protocol](../docs/small-data-controls.md) · [Raw report](small-layer-controls-01.json.gz)",
    "",
]
(root / "results/small-layer-controls.md").write_text("\n".join(lines))
