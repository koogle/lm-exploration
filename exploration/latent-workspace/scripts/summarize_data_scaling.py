"""Summarize completed size studies; validation selection is explicitly exploratory."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
rows = json.loads((ROOT / "data-scaling-summary.json").read_text())
lines = [
    "# Clean dataset-size study",
    "",
    "![Learning curves](data-scaling.png)",
    "",
    "Frozen Qwen2.5-0.5B, 23M-parameter diffusion head, four-token blocks, mixed",
    "prose/code/translation training. Identical 192 validation windows; test sealed.",
    "Separate position outputs and residual readout are enabled. These are",
    "all-masked marginal losses, not autoregressive sequence perplexities.",
    "",
    "| Run | Seed | Batch | Available unique windows | Best update | Best NLL | Final update | Final NLL |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['run']} | {row['seed']} | {row['batch']} | {row['windows']:,} | "
        f"{row['best_step']:,} | {row['best_validation_nll']:.3f} | "
        f"{row['final_step']:,} | {row['final_validation_nll']:.3f} |"
    )
lines += ["", "## Paired seed comparison at the 4k-update maximum budget", ""]
comparisons = []
for seed in (42, 43, 44):
    subset = [
        r for r in rows if r["seed"] == seed and r["batch"] == 128 and r["final_step"] == 4000
    ]
    small = next((r for r in subset if r["windows"] == 49152), None)
    large = next((r for r in subset if r["windows"] == 196608), None)
    if small and large:
        delta = large["best_validation_nll"] - small["best_validation_nll"]
        comparisons.append(delta)
        lines.append(
            f"- Seed {seed}: best observed larger-minus-smaller NLL {delta:.3f}; "
            f"selected updates {large['best_step']:,}/{small['best_step']:,}."
        )
if comparisons:
    lines += [
        "",
        (
            f"Mean paired difference: {sum(comparisons) / len(comparisons):.3f} "
            f"across {len(comparisons)} completed seeds. This summarizes training-seed "
            "variation on the same held-out data; it is not a population confidence interval."
        ),
    ]
lines += [
    "",
    "## Interpretation and limits",
    "",
    "The 49k pool overfits with continued training. Larger data supports longer",
    "optimization, but average loss improvements do not establish useful block",
    "generation. Inspect consecutive Qwen agreement in the right-hand plot.",
    "Batch comparisons change total token exposure, not only gradient noise.",
    "Available unique windows are not necessarily all sampled during a run.",
    "",
    "Roughly 200k windows is a diagnostic training pool, not a demonstrated",
    "minimum for acceptable quality. Use 4k batch-128 updates for initial decoder",
    "comparisons; extend promising changes selectively before expanding data.",
    "Best checkpoints are selected on the same",
    "small validation set used repeatedly for architecture and training choices.",
    "Larger validation and near-duplicate checks remain necessary before freezing",
    "a protocol and evaluating test. Human corpus labels are used; no target-model",
    "distillation, deployed verifier, quality parity or speedup is demonstrated.",
    "",
    "## Provenance",
    "",
    "Full reports retain configurations, source hashes, corpus revisions, window",
    "identities, training indices and per-example metrics. Large reports are",
    "losslessly archived as `RUN.json.gz`; others use `RUN.json`.",
    "Rebuild these figures and this summary:",
    "",
    "```bash",
    "uv run --with matplotlib python scripts/plot_data_scaling.py",
    "uv run python scripts/summarize_data_scaling.py",
    "```",
    "",
]
(ROOT / "data-scaling.md").write_text("\n".join(lines))
