"""Generate the research report and Matplotlib figures from measured candidates."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

COLORS = {16384: "#0072B2", 32768: "#D55E00", 50000: "#009E73"}
SIZES = [64, 128, 256, 512, 1024, 2048]
VOCABS = [16384, 32768, 50000]


def bootstrap(result, repetitions=256):
    """Parent bootstrap stratified by source; fixed source-byte weights.

    Identical RNG/parent ordering yields paired replicates across candidates.
    Intervals describe this sample, not collector-seed or downstream uncertainty.
    """
    rng = np.random.default_rng(20261016)
    by_source = {}
    for row in result["metrics"]["parent_totals"]:
        by_source.setdefault(row["source"], []).append(row)
    rates = np.zeros(repetitions)
    total = result["metrics"]["bytes"]
    for name in sorted(by_source):
        rows = sorted(by_source[name], key=lambda row: row["parent_sha256"])
        pairs = np.array([[r["bytes"], r["tokens"]] for r in rows], dtype=float)
        weight = pairs[:, 0].sum() / total
        # Bounded temporary memory even for many short held-out parents.
        for begin in range(0, repetitions, 16):
            indices = rng.integers(0, len(rows), size=(min(16, repetitions - begin), len(rows)))
            sampled = pairs[indices].sum(axis=1)
            rates[begin : begin + len(indices)] += weight * sampled[:, 1] / sampled[:, 0]
    return 1 / rates


def generate(folder):
    folder = Path(folder)
    raw = [json.loads(p.read_text()) for p in sorted(folder.glob("*mib-*.json"))]
    results = [r for r in raw if r.get("execution") == "modal-cpu"]
    if len({(r["size_mib"], r["requested_vocab_size"]) for r in results}) != len(results):
        raise ValueError("Duplicate candidate results")
    if len({r["corpus_manifest_sha256"] for r in results}) > 1:
        raise ValueError("Cannot compare candidates evaluated on different corpora")
    for field in (
        "config_sha256",
        "implementation_sha256",
        "tokenizers_version",
        "evaluation_manifest_sha256",
        "evaluation_implementation_sha256",
    ):
        if len({r.get(field) for r in results}) > 1:
            raise ValueError(f"Cannot compare candidates with different {field}")
    draws = {}
    compact = []
    for result in results:
        key = (result["size_mib"], result["requested_vocab_size"])
        if key[0] not in SIZES or key[1] not in VOCABS:
            raise ValueError("Unexpected candidate")
        draws[key] = bootstrap(result)
        summary = {
            **result,
            "metrics": {k: v for k, v in result["metrics"].items() if k != "parent_totals"},
        }
        summary["bytes_per_token_ci95"] = np.quantile(draws[key], [0.025, 0.975]).tolist()
        raw_path = folder / f"{key[0]}mib-{key[1]}.json"
        summary["raw_result_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        compact.append(summary)
    compact.sort(key=lambda r: (r["size_mib"], r["requested_vocab_size"]))
    by_key = {(r["size_mib"], r["requested_vocab_size"]): r for r in compact}
    for key, result in by_key.items():
        for label, base in [
            ("vs_64mib_same_vocab", (64, key[1])),
            ("vs_16k_same_size", (key[0], 16384)),
        ]:
            if base in by_key:
                # Same evaluation bytes: 1 - candidate token count / baseline token count.
                saving = 100 * (
                    1
                    - by_key[base]["metrics"]["bytes_per_token"]
                    / result["metrics"]["bytes_per_token"]
                )
                interval = np.quantile(100 * (1 - draws[base] / draws[key]), [0.025, 0.975])
                result[label] = {"token_saving_percent": saving, "paired_ci95": interval.tolist()}
    (folder / "measurements.json").write_text(json.dumps(compact, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "svg.hashsalt": "tokenizer-learning-curve-v1",
        }
    )
    figures = []

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", bbox_inches="tight")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        svg = folder / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        figures.append((name, fig))

    def decorate(ax, ylabel):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [str(s) for s in SIZES])
        ax.set_xlabel("Tokenizer fitting text (MiB)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    corpus_path = folder / "corpus-manifest.json"
    evaluation_path = folder / "evaluation-corpus-manifest.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    if results and evaluation is not None:
        expected = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        if any(r.get("evaluation_manifest_sha256") != expected for r in results):
            raise ValueError("Result does not match the published evaluation manifest")
    corpus_rows = []
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text())
        for name, source in corpus["sources"].items():
            held_out = evaluation["sources"][name] if evaluation else source
            corpus_rows.append(
                [
                    name,
                    str(source["weight_percent"]),
                    f"{source['train']['bytes'] / 1024**2:.2f}",
                    str(source["train"]["parents"]),
                    f"{held_out['validation']['bytes'] / 1024**2:.2f}",
                    str(held_out["validation"]["parents"]),
                ]
            )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        table = ax.table(
            cellText=corpus_rows,
            colLabels=["Source", "Weight %", "Fit MiB", "Fit parents", "Eval MiB", "Eval parents"],
            colWidths=[0.30, 0.12, 0.14, 0.14, 0.14, 0.14],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax.set_title(
            "Frozen corpus composition — largest fitting sample and shared evaluation", pad=16
        )
        save(fig, "corpus-composition")
    fig, ax = plt.subplots(figsize=(10, 5))
    for vocab in VOCABS:
        group = [r for r in compact if r["requested_vocab_size"] == vocab]
        if not group:
            continue
        x = [r["size_mib"] for r in group]
        y = [r["metrics"]["bytes_per_token"] for r in group]
        bounds = np.array([r["bytes_per_token_ci95"] for r in group])
        ax.plot(x, y, "o-", color=COLORS[vocab], label=f"{vocab:,} vocabulary")
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=COLORS[vocab], alpha=0.12)
    decorate(ax, "Held-out UTF-8 bytes / token (higher is better)")
    ax.set_title(f"Tokenizer learning curve — {len(compact)}/18 Modal candidates measured")
    if compact:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Awaiting measured candidates\nNo synthetic results plotted",
            transform=ax.transAxes,
            ha="center",
        )
    fig.tight_layout()
    save(fig, "learning-curve")
    if compact:
        names = list(compact[0]["metrics"]["sources"])
        fig, axes = plt.subplots(4, 3, figsize=(13, 14), constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            for vocab in VOCABS:
                group = [r for r in compact if r["requested_vocab_size"] == vocab]
                ax.plot(
                    [r["size_mib"] for r in group],
                    [r["metrics"]["sources"][name]["bytes_per_token"] for r in group],
                    "o-",
                    color=COLORS[vocab],
                    label=f"{vocab:,}",
                )
            decorate(ax, "Bytes / token")
            ax.set_title(name.replace("_", " "))
        axes.flat[0].legend()
        save(fig, "source-comparisons")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for ax, field, ylabel in zip(
            axes.flat,
            [
                "training_seconds",
                "peak_process_rss_mib",
                "vocabulary_use_fraction",
                "evaluation_mib_per_second",
            ],
            [
                "Fitting time (seconds)",
                "Peak process RSS (MiB)",
                "Held-out vocabulary use fraction",
                "Evaluation throughput (MiB/s)",
            ],
        ):
            for vocab in VOCABS:
                group = [r for r in compact if r["requested_vocab_size"] == vocab]
                ax.plot(
                    [r["size_mib"] for r in group],
                    [r[field] if field in r else r["metrics"][field] for r in group],
                    "o-",
                    color=COLORS[vocab],
                    label=f"{vocab:,}",
                )
            decorate(ax, ylabel)
        axes.flat[0].legend()
        save(fig, "resources-and-utilization")
        fig, ax = plt.subplots(figsize=(11, 5))
        for vocab in VOCABS:
            group = [
                r
                for r in compact
                if r["requested_vocab_size"] == vocab and "vs_64mib_same_vocab" in r
            ]
            ax.plot(
                [r["size_mib"] for r in group],
                [r["vs_64mib_same_vocab"]["token_saving_percent"] for r in group],
                "o-",
                color=COLORS[vocab],
                label=f"{vocab:,}",
            )
        decorate(ax, "Held-out token reduction vs 64 MiB (%)")
        ax.set_title("Benefit of increasing fitting data at a fixed vocabulary size")
        ax.legend()
        save(fig, "data-size-gains")
    shared_sizes = [size for size in SIZES if all((size, v) in by_key for v in VOCABS)]
    if shared_sizes:
        selected_size = max(shared_sizes)
        terms = list(by_key[(selected_size, VOCABS[0])]["metrics"]["fragmentation"])
        fig, ax = plt.subplots(figsize=(13, 5))
        locations = np.arange(len(terms))
        for index, vocab in enumerate(VOCABS):
            values = by_key[(selected_size, vocab)]["metrics"]["fragmentation"]
            ax.bar(
                locations + (index - 1) * 0.25,
                [values[t] for t in terms],
                width=0.25,
                color=COLORS[vocab],
                label=f"{vocab:,}",
            )
        ax.set_xticks(locations, terms, rotation=30, ha="right")
        ax.set_ylabel("Tokens per fixed probe (lower means less fragmentation)")
        ax.set_title(f"Technical, code and numeric probes — {selected_size} MiB fitting sample")
        ax.legend()
        fig.tight_layout()
        save(fig, "fragmentation-probes")
    matrix_complete = len(compact) == 18
    complete = matrix_complete and all(
        r["correctness_passed"] and r["requested_vocabulary_reached"] for r in compact
    )
    status = (
        "complete"
        if complete
        else "matrix complete — validation failed"
        if matrix_complete
        else "in progress — incomplete matrix"
    )
    overview = (
        f"{len(compact)} of 18 requested candidates have measured Modal results. "
        "The grid doubles fitting text from 64 MiB through 2 GiB and compares "
        "16,384, 32,768 and 50,000 vocabulary entries. All fresh candidates use "
        "the same source-weighted, parent-held-out evaluation corpus. "
        "A higher bytes-per-token value means fewer tokens for the same text; "
        "it does not establish better language-model quality."
    )
    findings = []
    if complete:
        best = max(compact, key=lambda r: r["metrics"]["bytes_per_token"])
        findings.append(
            f"The strongest observed compression is {best['metrics']['bytes_per_token']:.4f} "
            f"bytes/token at {best['size_mib']} MiB and {best['requested_vocab_size']:,} entries. "
            "This identifies a compression result, not a final model-optimal vocabulary."
        )
        for vocab in VOCABS:
            group = [by_key[(size, vocab)] for size in SIZES]
            start, penultimate, end = group[0], group[-2], group[-1]
            total_gain = 100 * (1 - end["metrics"]["tokens"] / start["metrics"]["tokens"])
            last_gain = 100 * (1 - end["metrics"]["tokens"] / penultimate["metrics"]["tokens"])
            best_tokens = min(r["metrics"]["tokens"] for r in group)
            near_best = min(
                r["size_mib"] for r in group if r["metrics"]["tokens"] <= 1.01 * best_tokens
            )
            findings.append(
                f"At {vocab:,} entries, 64 MiB → 2 GiB reduces held-out token count by "
                f"{total_gain:.2f}%; the final doubling contributes {last_gain:.3f}%. "
                f"The smallest sample within 1% of this vocabulary's best observed token "
                f"count is {near_best} MiB. The 1% tolerance is an analysis heuristic."
            )
        gain32 = by_key[(2048, 32768)]["vs_16k_same_size"]["token_saving_percent"]
        gain50 = by_key[(2048, 50000)]["vs_16k_same_size"]["token_saving_percent"]
        findings.append(
            f"At 2 GiB, 32K uses {gain32:.2f}% fewer tokens than 16K; 50K uses "
            f"{gain50:.2f}% fewer. At width 768, tied vocabulary parameters increase by "
            "12.58M and 25.82M respectively. Untied input/output matrices double these deltas."
        )
        findings.append(
            "The published default is the 2 GiB / 32K candidate, a practical starting point "
            "balancing compression and vocabulary parameter overhead. All 18 variants are "
            "published for comparison. This choice has not been validated by matched-compute "
            "language-model training."
        )
        overview += " " + findings[0]
    lines = [
        "# Tokenizer learning-curve research report",
        "",
        "## Overview",
        "",
        overview,
        "",
        f"**Status: {status}.**",
        "",
        "## Experimental design",
        "",
        (
            "- All 12 pinned source inventories from the Simple LLM plan are eligible; "
            "the experiment samples them rather than claiming to process every upstream record."
        ),
        (
            "- Nested per-source UTF-8 byte prefixes: 64, 128, 256, 512, 1,024 and 2,048 MiB. "
            "16K means 16,384 entries, 32K means 32,768, and 50K means exactly 50,000."
        ),
        (
            "- Source weights follow the plan, applied to bytes for tokenizer fitting. "
            "This is distinct from the eventual model-training token mixture."
        ),
        (
            "- Shared 32 MiB evaluation corpus; parent-level train/evaluation/test separation "
            "before chunking. Test parents are excluded. The final evaluation caps each document "
            "and parent at 16 KiB to broaden coverage; fitting data are unchanged."
        ),
        (
            "- Byte-level BPE, full byte alphabet, no normalization or automatic boundaries; "
            "the frozen v1 pretokenization rules are held fixed. The original tokenizer is unchanged."
        ),
        (
            "- Seeded shard permutation, sequential records within shards, a 64 KiB document cap "
            "and 256 KiB parent cap for fitting. Exact whitespace-normalized duplicate chunks are removed; "
            "cross-split conflicts are quarantined before freezing."
        ),
        (
            "- CPU-only Modal: eight-CPU and 64 GiB memory requests, with a two-hour candidate timeout; "
            "at most three candidates concurrently. Peak RSS includes fitting and evaluation. "
            "CPU/memory requests are not hard caps; Rayon uses eight threads. "
            "Final rescoring uses two CPUs and 8 GiB RAM (requests and hard caps), at most three "
            "workers and a 20-minute timeout. Throughput includes encoding, decoding, counting "
            "and checks; peak RSS refers to the original fitting/evaluation process."
        ),
        "",
        "## Evaluation coverage audit",
        "",
        (
            "The initial evaluation had only five LibreTexts parent books. Before interpreting "
            "the candidate curves, evaluation was rebuilt with a 16 KiB parent cap and the same "
            "32 MiB source quotas. Every candidate was rescored on that shared corpus. "
            "An initial 8 KiB cap could not fill the LibreTexts quota (636,017 bytes after "
            "exhausting the pinned inventory), so a fresh evaluation version used 16 KiB. "
            "Fitting text and tokenizer weights were preserved. Original fitting chunks were "
            "reserved in the dedup index to exclude exact normalized chunk overlap. Original "
            "result.json files remain on Modal; result-balanced-evaluation.json contains final metrics."
        )
        if evaluation
        else "Broader evaluation coverage is pending.",
        "",
        "## Aggregate learning curve",
        "",
        "![Learning curve](learning-curve.png)",
        "",
        (
            "Shading is a 95% parent-bootstrap interval, stratified by source with fixed source-byte "
            "weights (256 paired replicates). It describes evaluation-sample uncertainty, "
            "not uncertainty across training seeds."
        ),
        "",
        "| MiB | Vocabulary requested / actual | Bytes/token | Token reduction vs 64 MiB | Fit seconds | Peak RSS MiB | Correctness |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    if corpus_rows:
        insertion = lines.index("## Aggregate learning curve") - 1
        table_lines = [
            "",
            "## Frozen corpus composition",
            "",
            "| Source | Weight % | Fit MiB | Fit parents | Eval MiB | Eval parents |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        table_lines += ["| " + " | ".join(row) + " |" for row in corpus_rows]
        table_lines += [
            "",
            (
                "Parent counts use the adapter's source-scoped identifiers; "
                "equivalent works across sources are not fully reconciled."
            ),
            "",
        ]
        lines[insertion:insertion] = table_lines
    if findings:
        insertion = lines.index("## Experimental design") - 1
        lines[insertion:insertion] = ["", "## Main findings", "", *[f"- {f}" for f in findings], ""]
    for size in SIZES:
        for vocab in VOCABS:
            r = by_key.get((size, vocab))
            if r is None:
                lines.append(f"| {size} | {vocab:,} / — | pending | — | — | — | pending |")
                continue
            gain = r.get("vs_64mib_same_vocab", {}).get("token_saving_percent")
            gain_str = f"{gain:.2f}%" if gain is not None else "—"
            lines.append(
                f"| {size} | {vocab:,} / {r['actual_vocab_size']:,} | "
                f"{r['metrics']['bytes_per_token']:.4f} | {gain_str} | "
                f"{r['training_seconds']:.1f} | {r['peak_process_rss_mib']:.0f} | "
                f"{'pass' if r['correctness_passed'] else 'FAIL'} |"
            )
    if compact:
        lines += [
            "",
            "## Source-level comparisons",
            "",
            "![Source comparisons](source-comparisons.png)",
            "",
            "## Resource use and vocabulary utilization",
            "",
            "![Resource comparisons](resources-and-utilization.png)",
            "",
            (
                "Unseen vocabulary includes special/control entries. A low utilization fraction "
                "alone is not evidence of a poor tokenizer on this finite evaluation set."
            ),
            "",
            "## Gains from more fitting data",
            "",
            "![Data-size gains](data-size-gains.png)",
            "",
            "## Vocabulary-size tradeoffs",
            "",
            "| MiB | Vocabulary | Token reduction vs 16K | Paired 95% interval |",
            "|---:|---:|---:|---|",
        ]
        for r in compact:
            contrast = r.get("vs_16k_same_size")
            if contrast:
                lo, hi = contrast["paired_ci95"]
                lines.append(
                    f"| {r['size_mib']} | {r['requested_vocab_size']:,} | "
                    f"{contrast['token_saving_percent']:.2f}% | {lo:.2f}% to {hi:.2f}% |"
                )
        lines += [
            "",
            (
                "Larger vocabularies reduce sequence length but add embedding/output parameters. "
                "At width 768 with tied embeddings, 32K adds 12.58M parameters over 16K; "
                "50K adds 25.82M. Untied input/output weights double those increases. "
                "A final choice needs matched model-compute experiments."
            ),
        ]
    if shared_sizes:
        lines += [
            "",
            "## Fixed-term fragmentation",
            "",
            "![Fragmentation probes](fragmentation-probes.png)",
            "",
            (
                "These are diagnostic examples, not a representative task benchmark. "
                "Fewer pieces are not automatically better, particularly for numbers."
            ),
        ]
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        (
            "Do not select a winner from an incomplete matrix. Once complete, examine "
            "same-vocabulary gains from each doubling, source regressions, fragmentation probes "
            "and resource use together. A compression plateau is evidence of diminishing "
            "tokenizer-fitting returns, not proof of downstream optimality."
        ),
        "",
        (
            "This experiment uses one corpus sampling seed. It does not establish stability "
            "across independent corpus samples. Near duplicates with changed wording or shifted "
            "chunk boundaries and equivalent books/papers across sources may remain. These "
            "limitations must be resolved for strong generalization claims. The frozen v1 "
            "collector split is not reproducible from its handoff, so its historical results "
            "are not mixed into this matched learning curve."
        ),
        "",
        (
            "Token-level language-model perplexity is not compared here. Downstream evaluation "
            "should use the same held-out text and bits per byte under matched model size and "
            "compute. No model quality or billing-cost improvement is claimed."
        ),
        "",
        "## Reproduction and evidence",
        "",
        "- Published tokenizer variants: [Simple LLM tokenizer](https://huggingface.co/jfrick/scholarly-code-tokenizer)."
        if complete
        else "- Publication awaits the complete measured matrix.",
        (
            "- `measurements.json` contains compact metrics, per-source comparisons, fragmentation "
            "probes and raw-result hashes. Full parent totals and candidate tokenizers remain "
            "on the Modal results volume."
        ),
        (
            "- `corpus-manifest.json` records source pins, sampling settings, corpus hashes and "
            "collection decisions; `evaluation-corpus-manifest.json` pins the broader held-out corpus. "
            "`modal-smoke.json` records remote test execution."
        ),
        (
            "- Run `python tokenizer_report.py <results-folder>` to regenerate this report and "
            "Matplotlib PNG/SVG/PDF figures from downloaded raw result JSONs."
        ),
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines))
    with PdfPages(folder / "research-report.pdf") as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        cover.text(0.09, 0.92, "Tokenizer learning curve", fontsize=22, weight="bold")
        cover.text(0.09, 0.87, "Simple LLM pretraining • 2026-10-16", fontsize=13)
        paragraphs = [
            overview.split(" The strongest observed compression")[0],
            (
                "Design: six fitting sizes × three vocabulary sizes, nested samples "
                "from 12 pinned sources, and a shared 32 MiB held-out evaluation corpus."
            ),
            (
                "Metrics: bytes per token by source and overall; round trips and unknown tokens; "
                "vocabulary utilization and fixed-term fragmentation; fitting time and peak memory."
            ),
            (
                "Interpretation: compression is a proxy. Evaluate source regressions and compute "
                "tradeoffs before choosing a tokenizer. No downstream model-quality result is implied."
            ),
            (
                "Limitations: one sampling seed; no global fuzzy near-deduplication guarantee; "
                "cross-source parent reconciliation remains incomplete. The complete methodology "
                "and numeric comparisons are in REPORT.md and measurements.json."
            ),
        ]
        y = 0.80
        for paragraph in paragraphs:
            wrapped = textwrap.fill(paragraph, 83)
            cover.text(0.09, y, wrapped, fontsize=11, va="top", linespacing=1.5)
            y -= 0.035 * (wrapped.count("\n") + 1) + 0.045
        pdf.savefig(cover)
        plt.close(cover)
        if findings:
            findings_page = plt.figure(figsize=(8.27, 11.69))
            findings_page.text(0.09, 0.92, "Measured findings", fontsize=22, weight="bold")
            y = 0.83
            for finding in findings:
                wrapped = textwrap.fill(finding, 80)
                height = 0.027 * (wrapped.count("\n") + 1) + 0.055
                if y - height < 0.08:
                    pdf.savefig(findings_page)
                    plt.close(findings_page)
                    findings_page = plt.figure(figsize=(8.27, 11.69))
                    findings_page.text(
                        0.09, 0.92, "Measured findings (continued)", fontsize=20, weight="bold"
                    )
                    y = 0.83
                findings_page.text(0.09, y, wrapped, fontsize=11, va="top", linespacing=1.5)
                y -= height
            pdf.savefig(findings_page)
            plt.close(findings_page)
        for _, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    for _, fig in figures:
        plt.close(fig)
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    args = parser.parse_args()
    generate(args.folder)
