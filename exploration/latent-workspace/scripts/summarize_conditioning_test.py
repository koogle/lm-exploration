"""Report the frozen test suite, including paired source-group uncertainty."""

import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "latent-conditioning-test"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
protocol = json.loads((ROOT / "conditioning-test-protocol.json").read_text())
raw = json.loads((ROOT / "conditioning-test-01.json").read_text())["report"]
assert raw["evaluation_split"] == "test"
rows, studies = [], {}
for name, selected in protocol["checkpoints"].items():
    report = raw["studies"]["/runs/" + name]
    assert report["source_report_sha256"] == selected["source_report_sha256"]
    head = report["heads"]["196608"]
    assert head["step"] == selected["selected_step"]
    studies[name] = head
    domains = head["domains"]
    rows.append(
        {
            "run": name,
            "step": head["step"],
            "model": selected["model"],
            "validation_nll": selected["validation_nll"],
            "test_nll": sum(d["nll"] for d in domains.values()) / 3,
            "qwen_prefix": sum(d["qwen_greedy_prefix_agreement"] for d in domains.values()) / 3,
            "full_block_agreement": sum(
                d["qwen_greedy_full_block_agreement"] for d in domains.values()
            )
            / 3,
            "domain_nll": {d: v["nll"] for d, v in domains.items()},
        }
    )
comparisons = []
rng = np.random.default_rng(42)
for left, right in protocol["paired_comparisons"]:
    p = ROOT / f"{left}.json"
    if p.exists():
        source = json.loads(p.read_text())["report"]
    else:
        with gzip.open(str(p) + ".gz", "rt") as stream:
            source = json.load(stream)["report"]
    result = {"left": left, "right": right}
    for label, key in (("nll", "example_mean_nll"), ("qwen_prefix", "qwen_greedy_prefix_lengths")):
        estimates, draws = [], []
        for domain, (a, b) in source["data"]["domain_ranges"]["test"].items():
            delta = np.array(studies[left]["domains"][domain][key]) - np.array(
                studies[right]["domains"][domain][key]
            )
            groups = source["data"]["window_groups"]["test"][a:b]
            unique = sorted(set(groups))
            sums = np.array(
                [sum(v for v, g in zip(delta, groups) if g == group) for group in unique]
            )
            counts = np.array([groups.count(group) for group in unique])
            sample = rng.integers(len(unique), size=(5000, len(unique)))
            draws.append(sums[sample].sum(1) / counts[sample].sum(1))
            estimates.append(delta.mean())
        result[label] = {
            "left_minus_right": float(np.mean(estimates)),
            "group_bootstrap_95_interval": np.quantile(
                np.mean(draws, axis=0), [0.025, 0.975]
            ).tolist(),
        }
    comparisons.append(result)
summary = {
    "rows": rows,
    "paired_comparisons": comparisons,
    "limits": "One decoder seed, 192 grouped test windows, multiple exploratory comparisons. Intervals are conditional on trained heads; not corrected for multiple testing.",
}
(ROOT / "conditioning-test-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
names = {
    "conditioning-final-01": "Final residual",
    "conditioning-all-01": "All 24 residual layers",
    "conditioning-best-01": "Residual layer 18",
    "conditioning-selected-01": "Residual layers 18/20/17",
    "conditioning-attention-all-01": "All 24 attention outputs",
    "conditioning-attention-best-01": "Attention output 20",
    "conditioning-attention-selected-01": "Attention outputs 20/16/15",
    "conditioning-jspace-01": "J-lens 18/20/17",
    "conditioning-parallel-all-01": "Parallel, all residuals",
    "qwen3-conditioning-final-01": "Qwen3 0.6B, final",
    "qwen3-conditioning-all-01": "Qwen3 0.6B, all 28",
}
labels = [names[r["run"]] for r in rows]
y = np.arange(len(rows))
colors = ["#2779ad" if "Qwen2.5" in r["model"] else "#a86818" for r in rows]
bars = axes[0].barh(y, [r["test_nll"] for r in rows], color=colors)
axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
axes[0].set_xlim(0, 6.4)
bars = axes[1].barh(y, [r["qwen_prefix"] for r in rows], color=colors)
axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
for ax in axes:
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
axes[0].set_xlabel("Test all-masked NLL (lower is better)")
axes[1].set(xlabel="Consecutive Qwen matches / 4 (higher is better)", xlim=(0, 4))
fig.suptitle("Frozen-checkpoint test evaluation · 64 windows per domain · no test-based retuning")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(ROOT / "conditioning-test.png", dpi=170)
lines = [
    "# Frozen-checkpoint test evaluation",
    "",
    "![Test results](conditioning-test.png)",
    "",
    "All checkpoints were fixed by validation before test access. See",
    "[frozen protocol](conditioning-test-protocol.json) and full raw suite",
    "[report](conditioning-test-01.json). No test-based retuning was performed.",
    "",
    "| Run | Update | Test NLL | Prose | Code | Translation | Qwen prefix / 4 | Full agreement |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    domains = " | ".join(f"{row['domain_nll'][d]:.3f}" for d in ("prose", "code", "translation"))
    lines.append(
        f"| {row['run']} | {row['step']} | {row['test_nll']:.3f} | {domains} | "
        f"{row['qwen_prefix']:.3f} | {row['full_block_agreement']:.1%} |"
    )
lines += ["", "## Predeclared paired comparisons", ""]
for c in comparisons:
    nll = c["nll"]
    lo, hi = nll["group_bootstrap_95_interval"]
    lines.append(
        f"- {c['left']} minus {c['right']}: NLL {nll['left_minus_right']:.3f}, "
        f"source-group bootstrap interval [{lo:.3f}, {hi:.3f}]."
    )
lines += [
    "",
    summary["limits"],
    "",
    "All-masked NLL is not sequence perplexity. Qwen agreement is a diagnostic,",
    "not deployed speculative decoding or measured speedup. Model-family comparisons",
    "change architecture and projection parameter count as well as backbone size.",
    "",
    "## Saved test continuations",
    "",
]
for name, head in studies.items():
    lines += [f"### {name}", ""]
    for domain, text in head["samples"].items():
        lines += [
            f"**{domain}**",
            "",
            "Prefix:",
            "```text",
            head["sample_prefixes"][domain],
            "```",
            "Unanchored block continuation:",
            "```text",
            text,
            "```",
            "",
        ]
(ROOT / "conditioning-test.md").write_text("\n".join(lines))
print(json.dumps(summary, indent=2))
