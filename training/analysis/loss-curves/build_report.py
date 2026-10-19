"""Reproduce loss/optimizer plots from immutable copies of completed Modal logs."""

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASES = [
    ("Original mixed pretraining", "simple-llm-124m-8k-5b-20261018"),
    ("ECB / EMA continuation", "simple-llm-124m-eu-onepass-20261018"),
    ("60% code continuation", "simple-llm-code60-fresh-5b-20261019-r02"),
]


def read_phase(root, prefix):
    metrics, validation, provenance = {}, {}, []
    parent = None
    for folder in sorted(root.glob(prefix + "-s[0-9][0-9]")):
        report = json.loads((folder / "report.json").read_text())
        offset = report.get("lineage", {}).get("parent_tokens", 0)
        if parent is not None and parent != offset:
            raise ValueError("Parent token count changed within phase")
        parent = offset
        provenance.append(
            {
                "segment": folder.name,
                "report": report,
                "sha256": {
                    name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in ["metrics.jsonl", "validation.jsonl", "report.json"]
                },
            }
        )
        for filename, destination in [("metrics.jsonl", metrics), ("validation.jsonl", validation)]:
            for line in (folder / filename).read_text().splitlines():
                row = json.loads(line)
                row["lifetime_tokens"] = offset + row["tokens"]
                key = (row["step"], row["tokens"])
                if key in destination and filename == "metrics.jsonl":
                    raise ValueError("Replayed training update found")
                destination[key] = row
    if not metrics or not validation:
        raise ValueError(f"Missing training/validation logs: {prefix}")
    rows = sorted(metrics.values(), key=lambda r: r["tokens"])
    if any(b["tokens"] <= a["tokens"] for a, b in pairwise(rows)):
        raise ValueError("Nonmonotonic training token count")
    final = provenance[-1]["report"]
    if rows[-1]["tokens"] != final["total_loss_tokens"] or final["stop_reason"] != "token_limit":
        raise ValueError("Incomplete phase")
    return rows, sorted(validation.values(), key=lambda r: r["tokens"]), provenance


def binned(rows, field, width=128):
    points = []
    for i in range(0, len(rows), width):
        group = rows[i : i + width]
        weights = [r["loss_tokens"] for r in group]
        points.append(
            (
                group[-1]["lifetime_tokens"] / 1e9,
                np.average([r[field] for r in group], weights=weights),
            )
        )
    return np.array(points).T


def build(root, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(
        2, len(PHASES), figsize=(5 * len(PHASES), 8), constrained_layout=True, squeeze=False
    )
    optimizer, optaxes = plt.subplots(
        3, len(PHASES), figsize=(5 * len(PHASES), 10), constrained_layout=True, squeeze=False
    )
    provenance, summary, records, evaluations = [], [], [], []
    for column, (label, prefix) in enumerate(PHASES):
        rows, heldout, sources = read_phase(root, prefix)
        provenance.extend(sources)
        x = np.array([r["lifetime_tokens"] / 1e9 for r in rows])
        stride = max(1, len(rows) // 1800)
        axes[0, column].scatter(
            x[::stride],
            [r["loss"] for r in rows[::stride]],
            s=2,
            alpha=0.12,
            color="#4477aa",
            label="Individual updates (subsampled)",
        )
        bx, by = binned(rows, "loss")
        axes[0, column].plot(
            bx, by, color="#005a9c", lw=1.7, label="128-update token-weighted bins"
        )
        axes[0, column].set(title=label, ylabel="Training cross-entropy (nats/token)")
        for source in sorted(heldout[-1]["per_source"]):
            axes[1, column].plot(
                [r["lifetime_tokens"] / 1e9 for r in heldout],
                [r["per_source"][source] for r in heldout],
                alpha=0.55,
                lw=0.8,
            )
        axes[1, column].plot(
            [r["lifetime_tokens"] / 1e9 for r in heldout],
            [r["weighted_loss"] for r in heldout],
            color="black",
            lw=2,
            label="Mixture-weighted diagnostic",
        )
        axes[1, column].set(
            ylabel="Held-out validation cross-entropy",
            xlabel="Lifetime training tokens (billions)",
        )
        axes[1, column].legend(fontsize=8)
        axes[0, column].legend(fontsize=7)
        for rownum, (field, ylabel) in enumerate(
            [
                ("gradient_norm", "Gradient norm before clipping"),
                ("lr", "Learning rate"),
                ("throughput", "Nonpadding targets / second"),
            ]
        ):
            points = []
            for i in range(0, len(rows), 128):
                group = rows[i : i + 128]
                value = (
                    sum(r["loss_tokens"] for r in group) / sum(r["seconds"] for r in group)
                    if field == "throughput"
                    else np.mean([r[field] for r in group])
                )
                points.append((group[-1]["lifetime_tokens"] / 1e9, value))
            px, py = np.array(points).T
            optaxes[rownum, column].plot(px, py, color="#007766")
            optaxes[rownum, column].set(ylabel=ylabel)
            if rownum == 0:
                optaxes[rownum, column].set_title(label)
            if rownum == 2:
                optaxes[rownum, column].set_xlabel("Lifetime training tokens (billions)")
        for row in rows:
            records.append(
                {
                    "phase": prefix,
                    **{
                        k: row[k]
                        for k in [
                            "step",
                            "tokens",
                            "lifetime_tokens",
                            "loss_tokens",
                            "loss",
                            "lr",
                            "gradient_norm",
                            "seconds",
                        ]
                    },
                }
            )
        for row in heldout:
            for source, loss in row["per_source"].items():
                evaluations.append(
                    {
                        "phase": prefix,
                        "step": row["step"],
                        "tokens": row["tokens"],
                        "lifetime_tokens": row["lifetime_tokens"],
                        "source": source,
                        "loss": loss,
                        "weighted_loss": row["weighted_loss"],
                    }
                )
        summary.append(
            {
                "phase": prefix,
                "updates": len(rows),
                "final_phase_tokens": rows[-1]["tokens"],
                "final_lifetime_tokens": rows[-1]["lifetime_tokens"],
                "final_128_update_loss": float(by[-1]),
                "final_validation_diagnostic": heldout[-1]["weighted_loss"],
            }
        )
        for axis in [*axes[:, column], *optaxes[:, column]]:
            axis.grid(alpha=0.18)
    fig.suptitle(
        f"Simple LLM 124M — completed training through {summary[-1]['final_lifetime_tokens'] / 1e9:.3f}B tokens\n"
        "Different data mixtures and diagnostic windows across phases; losses are not directly comparable",
        fontsize=14,
    )
    optimizer.suptitle(
        "Simple LLM 124M — optimizer and training-step throughput\n"
        "128-update bins; throughput excludes checkpointing/evaluation and other job overhead",
        fontsize=14,
    )
    for name, figure in [("loss-curves", fig), ("optimizer-curves", optimizer)]:
        figure.savefig(output / (name + ".png"), dpi=160)
        figure.savefig(output / (name + ".pdf"))
        plt.close(figure)
    for name, data in [("training-metrics.csv", records), ("validation-metrics.csv", evaluations)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / "provenance.json").write_text(
        json.dumps({"segments": provenance, "summary": summary}, indent=2) + "\n"
    )
    representative = all(
        p["report"].get("validation_note") == "Frozen stratified held-out windows"
        for p in provenance
    )
    note = (
        "These curves use the frozen stratified held-out windows from this run. "
        "They are distinct from the larger final checkpoint evaluation."
        if representative
        else "Historical validation curves use the old one-window-per-source diagnostics, "
        "not the representative held-outs required for the new experiment."
    )
    (output / "README.md").write_text(
        "# Completed training curves\n\n![Loss curves](loss-curves.png)\n\n"
        "![Optimizer curves](optimizer-curves.png)\n\n"
        f"Lifetime training through {summary[-1]['final_lifetime_tokens']:,} targets. "
        "Only completed training phases are shown. Data preparation has no model loss.\n\n"
        + note
        + "\n\n"
        "Losses across different mixtures are not directly comparable. Training plots show "
        "128-update token-weighted bins plus a subsample of individual updates. "
        "Optimizer means use 128-update bins; throughput divides total targets by total "
        "training-step seconds and excludes checkpoint/evaluation/job overhead. "
        "CSV retains every logged update. Provenance records source-log hashes, reports "
        "and exact counts. Low loss does not establish factual accuracy, executable "
        "code quality or absence of repetition.\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.output)
