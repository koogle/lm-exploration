"""Plot completed clean size studies, preserving run/seed/batch provenance."""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "latent-workspace-data-scaling"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
names = [
    "data-repaired-02",
    "data-large-01",
    "data-batch128-01",
    "data-large-batch128-01",
    "data-continue-01",
    "data-seed43-01",
    "data-seed44-01",
]
reports = {}
for name in names:
    path = ROOT / "results" / f"{name}.json"
    if path.exists():
        value = json.loads(path.read_text())
    elif path.with_suffix(".json.gz").exists():
        with gzip.open(path.with_suffix(".json.gz"), "rt") as stream:
            value = json.load(stream)
    else:
        continue
    reports[name] = value.get("report", value)
reference = reports["data-repaired-02"]
for report in reports.values():
    for key in ("window_hashes", "window_groups", "domain_ranges"):
        assert report["data"][key]["validation"] == reference["data"][key]["validation"]
    for key in (
        "model",
        "model_revision",
        "layers",
        "selected_layers",
        "block_size",
        "width",
        "decoder_layers",
        "position_readout",
        "residual_readout",
        "learning_rate",
        "context_length",
        "memory_tokens",
        "slots",
        "refinement_rounds",
        "diffusion_steps",
    ):
        assert report["config"][key] == reference["config"][key], key
colors = {12288: "#a86818", 49152: "#2779ad", 196608: "#24804c"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
records = []
for name, report in reports.items():
    cfg = report["config"]
    for size, curve in report["curves"].items():
        rows = curve["checkpoints"]
        best = min(rows, key=lambda row: row["macro_validation_nll"])
        records.append(
            {
                "run": name,
                "seed": cfg["seed"],
                "batch": cfg["batch_size"],
                "windows": int(size),
                "best_step": best["step"],
                "best_validation_nll": best["macro_validation_nll"],
                "final_step": rows[-1]["step"],
                "final_validation_nll": rows[-1]["macro_validation_nll"],
            }
        )
        # The continuation contains the parent's checkpoints already.
        if name == "data-large-batch128-01" and "data-continue-01" in reports:
            continue
        steps = [row["step"] for row in rows]
        nll = [row["macro_validation_nll"] for row in rows]
        label = f"{int(size):,} windows"
        style = {42: "-", 43: "--", 44: ":"}[cfg["seed"]]
        panel = 0 if cfg["batch_size"] == 16 else 1
        if panel:
            label += f", seed {cfg['seed']}"
        axes[panel].plot(
            steps,
            nll,
            marker="o",
            markersize=3,
            linestyle=style,
            color=colors[int(size)],
            label=label,
        )
        if panel:
            agreement = [
                sum(v["qwen_greedy_prefix_agreement"] for v in row["validation_by_domain"].values())
                / 3
                for row in rows
            ]
            axes[2].plot(
                steps,
                agreement,
                marker="o",
                markersize=3,
                linestyle=style,
                color=colors[int(size)],
                label=label,
            )
for ax, title in zip(
    axes, ("Batch 16: dataset size", "Batch 128: size and training", "Batch 128: proposal quality")
):
    ax.set(xlabel="Optimizer updates", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
axes[0].set_ylabel("Validation all-masked NLL (lower is better)")
axes[1].set_ylabel("Validation all-masked NLL")
axes[2].set_ylabel("Mean consecutive Qwen matches / 4")
axes[2].set_ylim(0, 4)
fig.suptitle("Frozen Qwen2.5-0.5B + diffusion drafter · mixed training · test sealed")
fig.text(
    0.5,
    0.015,
    "Same 192 validation windows; repeated exploratory selection. Batch comparisons change token exposure.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
for suffix in ("png", "svg"):
    output = ROOT / "results" / f"data-scaling.{suffix}"
    fig.savefig(output, dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    if suffix == "svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
(ROOT / "results" / "data-scaling-summary.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
