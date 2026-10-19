"""Render aggregate validation evidence without exporting private token windows."""

import argparse
import json
from pathlib import Path

LABELS = {
    "original/fineweb_edu": "Educational web",
    "original/code_python": "Python code",
    "original/finemath_4plus": "Mathematics",
    "original/pes2o": "Scientific papers (peS2o)",
    "original/pubmed": "Biomedical papers (PubMed)",
    "original/arxiv": "arXiv",
    "original/pressbooks": "Pressbooks",
    "original/libretexts": "LibreTexts",
    "original/doab": "Open-access books (DOAB)",
    "original/gutenberg": "Gutenberg",
    "original/pre_1929_books": "Pre-1929 books",
    "original/library_of_congress": "Library of Congress",
    "ecb": "ECB",
    "ema": "EMA",
}


def build(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(Path(source).read_text())
    categories = data["categories"]
    names = [n for n in LABELS if n in categories]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = {k: v for k, v in categories.items() if k.startswith("original/")}
    worse = sum(v["direction"] == "worse" for v in original.values())
    sample = data["sample"]
    lines = [
        "# Expanded EU continuation validation",
        "",
        (
            f"Both checkpoints scored the same **{sample['total_tokens']:,} targets**, "
            f"**{sample['total_windows']:,} windows** and **{sample['total_families']:,} "
            "distinct source-family selections** across 14 categories. "
            "Full FP32; TF32 disabled; 2K context."
        ),
        "",
        (
            f"**{worse} of {len(original)} original categories have a positive "
            "after-minus-before loss difference with a category-wise 95% paired "
            "family-bootstrap interval above zero.**"
        ),
        "",
        "| Category | Before | After | Difference [95% CI] | Families | Targets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        r = categories[name]
        lo, hi = r["delta_ci95"]
        lines.append(
            f"| {LABELS[name]} | {r['before_loss']:.3f} | {r['after_loss']:.3f} | "
            f"{r['delta_loss']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            f"{r['families']:,} | {r['tokens']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "Losses are nats per token; lower is better. Differences are paired "
                "on identical windows. Intervals use 2,000 family-level resamples."
            ),
            "",
            "![Before/after loss and paired changes](expanded-validation.png)",
            "",
            "## Sampling and limits",
            "",
            sample["sampling_note"],
            "",
        ]
    )
    lines.extend("- " + note for note in data["limitations"])
    lines.extend(
        [
            "",
            (
                "Books with fewer than 256 validation families use all available families, "
                "with one sampled document per family. Original-category family identities "
                "come from the existing frozen split. The EU sample uses the continuation's "
                "document-family split; residual cross-language reconciliation remains pending."
            ),
            "",
            (
                "Absolute losses differ from the earlier small prefix diagnostic because "
                "this protocol changes document coverage, passage positions and precision."
            ),
            "",
            "## Provenance",
            "",
            "Original checkpoint: `simple-llm-124m-8k-5b-20261018-s03/step-000064227.pt`.",
            "",
            "EU checkpoint: `simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt`.",
            "",
            f"Original checkpoint SHA256: `{data['before_identity']['checkpoint_sha256']}`.",
            "",
            f"EU checkpoint SHA256: `{data['after_identity']['checkpoint_sha256']}`.",
            "",
            f"Shared sample SHA256: `{data['before_identity']['sample_sha256']}`.",
            "",
            (
                "Private source text, sampled token IDs and per-window losses remain on Modal. "
                "This folder contains aggregate evidence only. No training or HF writes."
            ),
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    y = np.arange(len(names))
    before = np.array([categories[n]["before_loss"] for n in names])
    after = np.array([categories[n]["after_loss"] for n in names])
    delta = after - before
    ci = np.array([categories[n]["delta_ci95"] for n in names])
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 8), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    left.barh(y - 0.16, before, height=0.3, color="#546a8a", label="Original 5B checkpoint")
    left.barh(y + 0.16, after, height=0.3, color="#d68a48", label="After EU continuation")
    left.set_yticks(y, [LABELS[n] for n in names])
    left.invert_yaxis()
    left.set_xlabel("Validation loss (nats/token; lower is better)")
    left.legend(loc="lower right", fontsize=8)
    for i, name in enumerate(names):
        color = {"better": "#218464", "worse": "#b14d49", "uncertain": "#777777"}[
            categories[name]["direction"]
        ]
        right.hlines(y[i], ci[i, 0], ci[i, 1], color=color)
        right.plot(ci[i], [y[i], y[i]], "|", color=color, markersize=7)
        right.plot(delta[i], y[i], "o", color=color)
    right.axvline(0, color="#666666", linewidth=1)
    right.set_xlabel("After − before loss (95% paired family-bootstrap CI)")
    for ax in (left, right):
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Simple LLM 124M: original 5B checkpoint vs +878.6M EU tokens", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"Identical {sample['total_tokens']:,} targets · "
        f"{sample['total_families']:,} source-family selections · FP32 · 2K windows",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output / "expanded-validation.png", dpi=180)
    fig.savefig(output / "expanded-validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", default=str(Path(__file__).parent))
    args = parser.parse_args()
    build(args.source, args.output)
