"""Render measured probe and trained decoder comparisons; never load test tensors."""

import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "latent-conditioning"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "conditioning"


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            value = json.load(stream)
    else:
        value = json.loads(path.read_text())
    return value.get("report", value) if isinstance(value, dict) else {}


def save(fig, name):
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(ROOT / f"{name}.png", dpi=170)
    path = ROOT / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


reports = {}
for path in sorted(ROOT.glob(f"{PREFIX}-*.json*")):
    report = load(path)
    name = path.name.removesuffix(".gz").removesuffix(".json")
    if report.get("completed"):
        reports[name] = report
scan = reports.get(f"{PREFIX}-scan-01")
if scan:
    replica = reports.get(f"{PREFIX}-scan-seed43-01")
    layers = scan["layer_scan"]["layers"]
    indices = scan["config"]["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for prefix, label in (("", "Residual stream"), ("attention", "Attention output")):
        line = axes[0].plot(
            indices,
            [layers[f"{prefix}{i}"]["future_nll"] for i in indices],
            marker="o",
            label=label + " (seed 42)",
        )[0]
        if replica:
            other = replica["layer_scan"]["layers"]
            axes[0].plot(
                indices,
                [other[f"{prefix}{i}"]["future_nll"] for i in indices],
                linestyle="--",
                color=line.get_color(),
                label=label + " (seed 43)",
            )
    for position in range(4):
        axes[1].plot(
            indices,
            [layers[str(i)]["nll_by_position"][position] for i in indices],
            marker=".",
            label=f"Target offset {position + 1}",
        )
    for ax in axes:
        ax.set(xlabel="Transformer block (one-based)", ylabel="Validation NLL")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title("Accessible future-token prediction: offsets 2–4")
    axes[1].set_title("Residual probes: target offsets (seed 42)")
    fig.suptitle("Frozen Qwen2.5-0.5B · 48 probes per seed · 500 updates each")
    fig.text(
        0.5,
        0.015,
        "Lower loss is better. Finite-budget probe evidence, not intrinsic information; validation only.",
        ha="center",
        fontsize=9,
    )
    save(fig, f"{PREFIX}-layer-scan")

runs = {name: r for name, r in reports.items() if r.get("curves") and "smoke" not in name}
if not runs:
    raise SystemExit("No completed decoder comparisons yet")
first = next(iter(runs.values()))
for report in runs.values():
    for key in ("window_hashes", "window_groups", "domain_ranges"):
        assert report["data"][key]["validation"] == first["data"][key]["validation"]
    for key in (
        "model",
        "model_revision",
        "layers",
        "position_readout",
        "residual_readout",
        "width",
        "batch_size",
        "learning_rate",
        "seed",
        "memory_tokens",
    ):
        assert report["config"][key] == first["config"][key], key
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
summary = []
for name, report in runs.items():
    cfg = report["config"]
    curve = report["curves"]["196608"]
    rows = curve["checkpoints"]
    label = name.removeprefix(PREFIX + "-").removesuffix("-01")
    steps = [row["step"] for row in rows]
    axes[0].plot(steps, [row["macro_validation_nll"] for row in rows], marker="o", label=label)
    agreement = [
        sum(v["qwen_greedy_prefix_agreement"] for v in row["validation_by_domain"].values()) / 3
        for row in rows
    ]
    axes[1].plot(steps, agreement, marker="o", label=label)
    best = min(rows, key=lambda row: row["macro_validation_nll"])
    domains = best["validation_by_domain"]
    summary.append(
        {
            "run": name,
            "variant": cfg["variants"][0],
            "selected_layers": cfg["selected_layers"],
            "best_step": best["step"],
            "best_nll": best["macro_validation_nll"],
            "qwen_prefix_agreement": sum(
                v["qwen_greedy_prefix_agreement"] for v in domains.values()
            )
            / 3,
            "domains": {d: v["nll"] for d, v in domains.items()},
            "parameters": curve["trainable_parameters"],
        }
    )
for ax in axes:
    ax.set_xlabel("Optimizer updates")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
axes[0].set(ylabel="Validation all-masked NLL", title="Trained conditioning comparison")
axes[1].set(
    ylabel="Consecutive Qwen matches / 4", ylim=(0, 4), title="Actual iterative block proposals"
)
fig.suptitle(f"{first['config']['model']} · 196,608 mixed-domain windows · matched training budget")
fig.text(
    0.5,
    0.015,
    "Common final-state residual readout; one decoder seed; test withheld during selection.",
    ha="center",
    fontsize=9,
)
save(fig, f"{PREFIX}-comparison")
(ROOT / f"{PREFIX}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Trained conditioning comparison",
    "",
    f"![Layer probes]({PREFIX}-layer-scan.png)"
    if scan
    else "Layer scan not part of this model comparison.",
    "",
    f"![Decoder training]({PREFIX}-comparison.png)",
    "",
    "Same 196k training windows and 192 validation windows; test withheld during selection.",
    "Decoder comparisons share a final-state residual shortcut. Probe ranking",
    "uses no such shortcut. All-masked loss is not sequence likelihood.",
    "",
    "| Condition | Best update | NLL | Prose | Code | Translation | Qwen prefix / 4 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in sorted(summary, key=lambda r: r["best_nll"]):
    domain = " | ".join(f"{row['domains'][d]:.3f}" for d in ("prose", "code", "translation"))
    lines.append(
        f"| {row['run']} | {row['best_step']} | {row['best_nll']:.3f} | "
        f"{domain} | {row['qwen_prefix_agreement']:.3f} |"
    )
lines += [
    "",
    "Best checkpoints are selected by validation NLL within each run.",
    "This exploratory ranking does not establish generalization to new source",
    "groups, significance across many layer choices, or useful generation quality.",
    "Raw JSON/JSON.gz preserves model/corpus/source revisions, per-example metrics",
    "and checkpoint provenance. See ../docs/conditioning-study.md for the protocol.",
    "",
]
(ROOT / f"{PREFIX}-report.md").write_text("\n".join(lines))
