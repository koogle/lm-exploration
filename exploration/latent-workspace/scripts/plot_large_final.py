"""Archive and plot the completed large-data curriculum; test remains sealed."""

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

reports = {
    cap: json.loads(Path(f"artifacts/large-r{cap}-final.json").read_text()) for cap in (64, 256)
}
a, b = reports.values()
assert a["initial_drafter_sha256"] == b["initial_drafter_sha256"]
assert a["source_sha256"] == b["source_sha256"]
for split in ("validation", "validation_short"):
    assert a["split_sequence_sha256"][split] == b["split_sequence_sha256"][split]
assert all(
    r["training_completed"] and r["verification_passed"] and r["curve"][-1]["step"] == 100000
    for r in reports.values()
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for col, (cap, report) in enumerate(reports.items()):
    curve = [c for c in report["curve"] if c["step"]]
    steps = [c["step"] / 1000 for c in curve]
    switch = report["curriculum_transition"]["step"] / 1000
    ax = axes[0, col]
    for d in ("prose", "code", "translation"):
        ax.plot(
            steps,
            [c["validation"]["by_domain"][d]["mean_accepted_beyond_anchor"] for c in curve],
            label=d,
            alpha=0.8,
        )
    ax.plot(
        steps,
        [c["validation"]["accepted_beyond_anchor"] for c in curve],
        color="black",
        linewidth=2,
        label="Overall",
    )
    best = next(c for c in curve if c["step"] == report["best_step"])
    x, y = best["step"] / 1000, best["validation"]["accepted_beyond_anchor"]
    ax.scatter([x], [y], color="purple", s=70, zorder=5)
    ax.annotate(
        f"Best {y:.2f} at {x:.1f}K", (x, y), xytext=(8, 12), textcoords="offset points", fontsize=9
    )
    ax.set(
        title=f"{cap}-token response cap",
        ylabel="Validation consecutive correct NEW tokens / 7",
        ylim=(0, 7),
    )
    ax = axes[1, col]
    ax.plot(steps, [c["validation"]["nll"] for c in curve], label="Validation: 192 responses")
    trained = [c for c in curve if c["step"] / 1000 > switch]
    ax.plot(
        [c["step"] / 1000 for c in trained],
        [c["training_diagnostic"]["nll"] for c in trained],
        linestyle="--",
        label="Training diagnostic: 12 responses",
    )
    ax.set(
        ylabel="All-domain NLL (lower is better)",
        xlabel="Optimizer updates (thousands)",
        ylim=(0, 8),
    )
    for ax in axes[:, col]:
        ax.axvline(switch, color="grey", linestyle=":", label="Translation added")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
fig.suptitle("Both large-data curriculum runs completed 100,000 updates")
fig.text(
    0.5,
    0.01,
    "131K prose/code → 196K balanced data; transition timing differs. Tiny training diagnostic; sealed test not evaluated.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
base = Path("results/large-curriculum-final")
fig.savefig(base.with_suffix(".png"), dpi=150)
base.with_suffix(".json").write_text(
    json.dumps(
        {
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "status": "TRAINING_COMPLETE_QUALIFICATION_PENDING",
            "reports": reports,
        },
        indent=2,
    )
    + "\n"
)
