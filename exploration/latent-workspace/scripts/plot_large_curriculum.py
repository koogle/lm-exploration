"""Archive the completed prose/code stage and plot matched-domain learning curves."""

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

reports = {
    cap: json.loads(Path(f"artifacts/large-r{cap}-01-live.json").read_text()) for cap in (64, 256)
}
a, b = reports.values()
assert a["initial_drafter_sha256"] == b["initial_drafter_sha256"]
assert a["source_sha256"] == b["source_sha256"]
assert a["split_sequence_sha256"]["validation"] == b["split_sequence_sha256"]["validation"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for col, (cap, report) in enumerate(reports.items()):
    curve = [c for c in report["curve"] if c["step"]]
    steps = [c["step"] / 1000 for c in curve]
    ax = axes[0, col]
    ax.plot(
        steps,
        [sum(c["validation"]["by_domain"][d]["nll"] for d in ("prose", "code")) / 2 for c in curve],
        marker="o",
        label="Prose/code validation: 128 responses",
    )
    ax.plot(
        steps,
        [c["training_diagnostic"]["nll"] for c in curve],
        linestyle="--",
        label="Prose/code training diagnostic: 8 responses",
    )
    ax.plot(
        steps,
        [c["validation"]["nll"] for c in curve],
        linestyle=":",
        color="grey",
        label="All-domain validation (translation not trained yet)",
    )
    ax.set(
        title=f"{cap}-token cap",
        xlabel="Optimizer updates (thousands)",
        ylabel="New-token NLL (lower is better)",
        ylim=(0, 8),
    )
    ax = axes[1, col]
    for domain in ("prose", "code", "translation"):
        ax.plot(
            steps,
            [c["validation"]["by_domain"][domain]["mean_accepted_beyond_anchor"] for c in curve],
            marker="o",
            label=domain,
        )
    ax.set(
        xlabel="Optimizer updates (thousands)",
        ylabel="Validation consecutive correct NEW tokens / 7",
        ylim=(0, 7),
    )
for ax in axes.flat:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("Large-data curriculum: first stage uses 65,536 prose + 65,536 code examples")
fig.text(
    0.5,
    0.01,
    "Frozen Qwen2.5-0.5B + fresh 79M drafter. Translation joins next stage. Training diagnostic is not a full-corpus audit.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
base = Path("results/large-curriculum-prose-code")
fig.savefig(base.with_suffix(".png"), dpi=150)
base.with_suffix(".json").write_text(
    json.dumps(
        {
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "stage": "PROSE_CODE_ONLY",
            "study_status": "IN_PROGRESS",
            "reports": reports,
        },
        indent=2,
    )
    + "\n"
)
