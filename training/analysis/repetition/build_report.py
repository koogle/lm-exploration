"""Build a checkpoint comparison report and plot from measured sampling results."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path("../../artifacts/repetition"),
        help="Folder with the sampling results; the report is written beside them",
    )
    folder = parser.parse_args().folder

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((folder / "low-temperature-20261018.json").read_text())
    previous = json.loads((folder / "previous-comparison-20261018.json").read_text())
    old = {c["step"]: c for c in previous["checkpoints"]}
    rows = []
    matches = 0
    for checkpoint in data["checkpoints"]:
        baseline = old[checkpoint["step"]]
        for sample in checkpoint["samples"]:
            if sample["temperature"] == 0:
                reference = next(
                    s
                    for s in baseline["samples"]
                    if s["mode"] == "greedy" and s["prompt"] == sample["prompt"]
                )
                matches += sample["generation_ids"] == reference["generation_ids"]
        scores = checkpoint["decoding_summary"]
        rows.append(
            {
                "step": checkpoint["step"],
                "tokens": checkpoint["trained_tokens"],
                "heldout_loss": baseline["heldout"]["mixture_weighted_loss"],
                "greedy": scores["0.0"]["mean_repeated_4gram_fraction"],
                "temperature_01": scores["0.1"]["mean_repeated_4gram_fraction"],
                "temperature_02": scores["0.2"]["mean_repeated_4gram_fraction"],
                "temperature_04": scores["0.4"]["mean_repeated_4gram_fraction"],
                "temperature_08_previous": sum(
                    baseline["decoding_summary"][m]["mean_repeated_4gram_fraction"]
                    for m in ["sampled_08_seed1", "sampled_08_seed2"]
                )
                / 2,
            }
        )
    assert len(data["checkpoints"]) == 4
    assert sum(len(c["samples"]) for c in data["checkpoints"]) == 280
    summary = {
        "rows": rows,
        "completions": 280,
        "greedy_completions_identical_to_previous_run": matches,
        "previous_greedy_completions": 40,
        "top_k": 50,
        "temperatures": [0.1, 0.2, 0.4],
        "sampling_seeds_per_prompt": 2,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = [r["tokens"] / 1e9 for r in rows]
    for key, label, color in [
        ("greedy", "Greedy", "#264c78"),
        ("temperature_01", "T=0.1, top-k=50", "#a4413d"),
        ("temperature_02", "T=0.2, top-k=50", "#c8862b"),
        ("temperature_04", "T=0.4, top-k=50", "#7756a3"),
        ("temperature_08_previous", "T=0.8, top-k=50 (previous study)", "#218b78"),
    ]:
        ax.plot(x, [r[key] * 100 for r in rows], "-o", label=label, color=color)
    ax.set(
        xlabel="Training exposures (billions of tokens)",
        ylabel="Repeated generated token 4-grams (%)",
        ylim=(0, 100),
        xlim=(0, 5.1),
    )
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle(
        "Simple LLM 124M: top-k sampling at low temperatures", fontsize=16, fontweight="bold"
    )
    fig.text(
        0.09,
        0.02,
        "Ten fixed prompts; two seeds per sampled setting; 128-token cap. Repetition is not a factual-accuracy metric.",
        fontsize=9,
        color="#526171",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(folder / "low-temperature.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / "low-temperature.pdf", bbox_inches="tight")
    lines = [
        "# Simple LLM 124M: repetition and low-temperature sampling",
        "",
        "This standalone study tests published checkpoints without training updates. It loads the exact model definition and weights from pinned Hugging Face revisions rather than the current repository architecture.",
        "",
        "## Results",
        "",
        "Repeated token four-gram fraction (1 minus unique token four-grams / all token four-grams), averaged over the same ten prompts. Sampled settings use two seeds per prompt. Lower repetition does not establish factual correctness.",
        "",
        "| Trained tokens | Held-out loss¹ | Greedy | T=0.1 | T=0.2 | T=0.4 | T=0.8¹ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tokens'] / 1e9:.3f}B | {r['heldout_loss']:.4f} | {r['greedy']:.1%} | {r['temperature_01']:.1%} | {r['temperature_02']:.1%} | {r['temperature_04']:.1%} | {r['temperature_08_previous']:.1%} |"
        )
    lines += [
        "",
        "¹ Held-out loss and T=0.8 results are retained from the previous matched-checkpoint study. That study used 1,798,432 identical targets from 384 validation parents, at most four 2K windows per parent. Pending corpus audits mean this is exploratory evaluation, not a sealed benchmark. Its final loss of 1.9635 differs from the training-time 1.8708 diagnostic because the samples differ.",
        "",
        "![Low-temperature comparison](low-temperature.png)",
        "",
        "Very low temperature did not resolve repetition: T=0.1 and T=0.2 stayed close to greedy decoding across all checkpoints. T=0.4 reduced repetition but remained substantially more repetitive than the previous T=0.8 setting.\n\nTop-k is fixed at 50 for all sampled settings. Temperatures rescale logits before sampling; lower temperatures concentrate probability on the highest-scoring candidates. No repetition penalty or n-gram ban is used in this study. All 280 completions are retained.",
        "",
        f"Greedy reproducibility: **{matches}/40** completions match the prior study token for token. Both experiments used PyTorch 2.6, H100, BF16 autocast with FP32 weights and the same ten prompts.",
        "",
        "## Training checks retained from the earlier study",
        "",
        "280 real training windows passed exact next-token alignment, EOS, document-boundary and padding-label checks, including 21 long-document continuations. Cursor replay reproduced the next batch. Twelve raw-loader, prefetch/resume and independent architecture-reference tests passed. A two-rank FP32 CPU/Gloo reference checked the production accumulation gradients and AdamW update. These checks found no simple implementation error; they do not prove the entire training/data pipeline is free of bugs.",
        "",
        "The previous study, including source audits, was produced at source commit `64ecf8ef56a649cc04a500eced3d03a052807799`. See [previous comparison](previous-comparison-20261018.json), [data audit](data-audit-20261018.json), and [implementation checks](implementation-checks-20261018.json).",
        "",
        "## Reproduce",
        "",
        "Run from `training/analysis/repetition/` in the lm-exploration repository with Modal credentials and access to the project output volume:",
        "",
        "```sh",
        "modal run --detach modal_low_temperature.py",
        "python build_report.py ../../artifacts/repetition",
        "```",
        "",
        "The remote job uses one H100 and a 30-minute hard timeout with no automatic function retry. Checkpoint downloads have four bounded retry attempts, and reruns reuse matching completed checkpoint results from the output volume. It verifies checkpoint weight hashes, model source identity, parameter count and finite parameters. Published checkpoint code is used only from our own pinned revisions. Deploying/importing this file does not launch training. The report builder additionally needs matplotlib.",
        "",
        "## Files",
        "",
        "[Raw low-temperature results](low-temperature-20261018.json) · [All completions](all-completions.md) · [Summary](summary.json) · [PDF plot](low-temperature.pdf)",
        "",
    ]
    (folder / "README.md").write_text("\n".join(lines))
    samples = ["# All low-temperature completions", ""]
    for c in data["checkpoints"]:
        samples += [f"## Step {c['step']:,}: {c['trained_tokens']:,} trained tokens", ""]
        for s in c["samples"]:
            samples += [
                f"### {s['mode']} / seed {s['seed']}",
                "",
                "Prompt:",
                "~~~text",
                s["prompt"],
                "~~~",
                "",
                "Continuation:",
                "~~~text",
                s["completion"],
                "~~~",
                "",
            ]
    (folder / "all-completions.md").write_text("\n".join(samples))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
