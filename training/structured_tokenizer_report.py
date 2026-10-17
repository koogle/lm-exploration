"""Matplotlib report for measured structured-output tokenizer experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from structured_tokenizer import CATEGORIES

ROOT = Path("results/tokenizer-structured/structured-code-v1")
VOCABS = (16384, 32768, 50000)


def paired(old, new):
    a = sorted(old["per_parent"], key=lambda r: r["parent_sha256"])
    b = sorted(new["per_parent"], key=lambda r: r["parent_sha256"])
    if [(r["parent_sha256"], r["bytes"]) for r in a] != [
        (r["parent_sha256"], r["bytes"]) for r in b
    ]:
        raise ValueError("Unpaired benchmark rows")
    rng = np.random.default_rng(20261016)
    ix = rng.integers(len(a), size=(1000, len(a)))
    x = np.array([r["tokens"] for r in a])[ix].sum(axis=1)
    y = np.array([r["tokens"] for r in b])[ix].sum(axis=1)
    return {
        "token_saving_percent": 100 * (1 - new["tokens"] / old["tokens"]),
        "paired_ci95": np.quantile(100 * (1 - y / x), [0.025, 0.975]).tolist(),
    }


def generate():
    raw = json.loads(Path("runs/structured-code-v1/scores.json").read_text())
    expected = {("original", s, v) for s in [64, 128, 256, 512, 1024, 2048] for v in VOCABS} | {
        ("focused", 64, v) for v in VOCABS
    }
    by = {(r["family"], r["size_mib"], r["vocab"]): r for r in raw}
    if set(by) != expected or len(raw) != 21 or len({r["evaluation_sha256"] for r in raw}) != 1:
        raise ValueError("Incomplete or incompatible measurements")
    if not all(r["correctness_passed"] for r in raw):
        raise ValueError("Correctness failure")
    fits = json.loads((ROOT / "focused-fits.json").read_text())
    baseline = json.loads(
        Path(
            "results/tokenizer-learning-curve/learning-curve-20261016-v2/measurements.json"
        ).read_text()
    )
    broad = {(r["size_mib"], r["requested_vocab_size"]): r for r in baseline}
    manifest = json.loads((ROOT / "corpus-manifest.json").read_text())
    if any(r["evaluation_sha256"] != manifest["evaluation_sha256"] for r in raw):
        raise ValueError("Benchmark manifest mismatch")
    fit_by = {r["vocab"]: r for r in fits}
    for r in raw:
        reference = (
            broad[(r["size_mib"], r["vocab"])] if r["family"] == "original" else fit_by[r["vocab"]]
        )
        if r["tokenizer_sha256"] != reference["tokenizer_sha256"]:
            raise ValueError("Candidate artifact mismatch")
    contrasts = []
    for v in VOCABS:
        for c in CATEGORIES:
            for size in (64, 2048):
                contrasts.append(
                    {
                        "vocab": v,
                        "category": c,
                        "baseline_size_mib": size,
                        **paired(
                            by[("original", size, v)]["metrics"][c],
                            by[("focused", 64, v)]["metrics"][c],
                        ),
                    }
                )
    tool_kinds = ("read_file_call", "apply_patch_call", "tool_result", "schema_and_search_call")
    tool_breakdown = []
    for r in raw:
        groups = []
        for kind, label in enumerate(tool_kinds):
            parents = [
                p
                for p in r["metrics"]["tool_call"]["per_parent"]
                if int(p["parent_sha256"][:8], 16) % 4 == kind
            ]
            groups.append(
                {
                    "name": r["name"],
                    "kind": label,
                    "examples": len(parents),
                    "bytes": sum(p["bytes"] for p in parents),
                    "tokens": sum(p["tokens"] for p in parents),
                    "mean_tokens_per_example": sum(p["tokens"] for p in parents) / len(parents),
                }
            )
        if sum(g["tokens"] for g in groups) != r["metrics"]["tool_call"]["tokens"]:
            raise ValueError("Tool subtype totals do not reconcile")
        tool_breakdown.extend(groups)
    (ROOT / "tool-subtypes.json").write_text(json.dumps(tool_breakdown, indent=2) + "\n")
    summaries = []
    for r in sorted(raw, key=lambda r: (r["family"], r["size_mib"], r["vocab"])):
        summaries.append(
            {
                **r,
                "metrics": {
                    c: {k: v for k, v in m.items() if k != "per_parent"}
                    for c, m in r["metrics"].items()
                },
            }
        )
    (ROOT / "measurements.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (ROOT / "paired-comparisons.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "structured-code-v1"}
    )
    figs = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(ROOT / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(ROOT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        path = ROOT / f"{name}.svg"
        path.write_text("\n".join(l.rstrip() for l in path.read_text().splitlines()) + "\n")
        figs.append(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, c in zip(axes.flat, CATEGORIES, strict=True):
        for v, color in zip(VOCABS, ["#0072B2", "#D55E00", "#009E73"], strict=True):
            group = sorted(
                [r for r in raw if r["family"] == "original" and r["vocab"] == v],
                key=lambda r: r["size_mib"],
            )
            ax.plot(
                [r["size_mib"] for r in group],
                [r["metrics"][c]["bytes_per_token"] for r in group],
                "-o",
                label=f"{v:,} original",
                color=color,
            )
            y = by[("focused", 64, v)]["metrics"][c]["bytes_per_token"]
            ax.scatter([64], [y], marker="*", s=150, color=color)
            ax.axhline(y, color=color, linestyle=":", alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512, 1024, 2048], [64, 128, 256, 512, 1024, 2048])
        ax.set(title=c, xlabel="Fitting MiB (star = new focused 64 MiB)", ylabel="Bytes/token ↑")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    save(fig, "structured-curves")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, size in zip(axes, [64, 2048], strict=True):
        for j, v in enumerate(VOCABS):
            rows = [
                next(
                    x
                    for x in contrasts
                    if x["vocab"] == v and x["category"] == c and x["baseline_size_mib"] == size
                )
                for c in CATEGORIES
            ]
            y = np.array([r["token_saving_percent"] for r in rows])
            ci = np.array([r["paired_ci95"] for r in rows])
            ax.bar(np.arange(4) + (j - 1) * 0.25, y, width=0.24, label=f"{v:,}")
            ax.errorbar(
                np.arange(4) + (j - 1) * 0.25,
                y,
                yerr=np.maximum(0, np.array([y - ci[:, 0], ci[:, 1] - y])),
                fmt="none",
                color="black",
                capsize=3,
            )
        ax.set_xticks(range(4), CATEGORIES)
        ax.set(title=f"Focused 64 MiB vs original {size} MiB", ylabel="Token reduction (%) ↑")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.legend()
    save(fig, "focused-gains")
    general = []
    for fit in fits:
        v = fit["vocab"]
        for size in (64, 2048):
            old = broad[(size, v)]
            if old["evaluation_manifest_sha256"] != fit["general_evaluation_sha256"]:
                raise ValueError("General evaluation mismatch")
            general.append(
                {
                    "vocab": v,
                    "baseline_size_mib": size,
                    "token_change_percent": 100
                    * (fit["general_metrics"]["tokens"] / old["metrics"]["tokens"] - 1),
                    "bytes_per_token": fit["general_metrics"]["bytes_per_token"],
                }
            )
    (ROOT / "general-regression.json").write_text(json.dumps(general, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, size in enumerate((64, 2048)):
        vals = [
            next(
                r["token_change_percent"]
                for r in general
                if r["vocab"] == v and r["baseline_size_mib"] == size
            )
            for v in VOCABS
        ]
        ax.bar(np.arange(3) + (j - 0.5) * 0.35, vals, width=0.34, label=f"vs original {size} MiB")
    ax.set_xticks(range(3), [f"{v:,}" for v in VOCABS])
    ax.set(
        xlabel="Vocabulary",
        ylabel="Extra tokens on scholarly mixture (%) ↓",
        title="Cost of code-focused fitting on the original 12-source evaluation",
    )
    ax.legend()
    save(fig, "general-regression")
    manifest = json.loads((ROOT / "corpus-manifest.json").read_text())
    lines = [
        "# JSON, code, Git diff and tool-call tokenizer study",
        "",
        "## Overview",
        "",
        "All 18 original tokenizers and three fresh code-focused tokenizers were scored on Modal. The new fits use approximately 64 MiB each at 16,384, 32,768 and 50,000 vocabulary entries. Existing tokenizers and the published default are unchanged.",
        "",
        "## Method",
        "",
        "- 512 held-out Python parent identifiers, one source excerpt per parent, sampled with seed 20261016 from the prior frozen evaluation. Each supplies four benchmark examples (2,048 total).",
        "- Python uses real code excerpts. JSON, Git diffs and tool calls are generated controlled fixtures derived from those excerpts. They are not a representative collection of production tool traces.",
        "- JSON includes nested metadata, booleans/null/numbers, compact/pretty formatting, escaped code and Unicode. Tool formats include schemas, argument objects, JSON-encoded arguments, calls and results.",
        "- Unified patches include insertions, deletions and identifier replacements. Generated fixture patches pass git apply checks in tests; binary patches, renames and multi-file edits are not covered.",
        "- The focused fitting mixture is 40% Python, 20% JSON, 20% diffs, 20% tool formats by UTF-8 bytes. Whole examples may slightly exceed each quota. Fitting parents are disjoint from evaluation parents; generator templates are shared.",
        "- Same byte-level BPE, pretokenization, full byte alphabet, four special tokens, min frequency 2 and maximum learned token length 32 as the original experiment. No 120K experiment.",
        "- Paired 95% intervals use 1,000 bootstrap samples of source parents. They quantify variation within this fixture set, not production generalization or training-seed variance.",
        f"- Actual fitting bytes: {sum(manifest['actual_bytes'].values()):,}; fitting source parents: {manifest['fitting_parents']:,}.",
        "- All 21 structured evaluations passed exact round-trip and zero-UNK checks. The three new fits were also evaluated on the original shared 32 MiB scholarly mixture.",
        "",
        "## Focused fits versus matched 64 MiB originals",
        "",
        "| Vocabulary | Python token savings | JSON | Git diff | Tool call |",
        "|---:|---:|---:|---:|---:|",
    ]
    for v in VOCABS:
        vals = [
            next(
                r["token_saving_percent"]
                for r in contrasts
                if r["vocab"] == v and r["category"] == c and r["baseline_size_mib"] == 64
            )
            for c in CATEGORIES
        ]
        lines.append("| " + str(v) + " | " + " | ".join(f"{x:.2f}%" for x in vals) + " |")
    lines += [
        "",
        "Positive savings mean fewer tokens; negative values mean more. Comparisons are within each workload, not an aggregate dominated by long code strings.",
        "",
        "![Gains](focused-gains.png)",
        "",
        "## All 21 candidates",
        "",
        "![Curves](structured-curves.png)",
        "",
        "| Candidate | Python bytes/token | JSON | Git diff | Tool call |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(f"{r['metrics'][c]['bytes_per_token']:.4f}" for c in CATEGORIES)
            + " |"
        )
    lines += [
        "",
        "## General-text regression check",
        "",
        "![General regression](general-regression.png)",
        "",
        "| Vocabulary | New bytes/token | Extra tokens vs original 64 MiB | Extra tokens vs original 2 GiB |",
        "|---:|---:|---:|---:|",
    ]
    for v in VOCABS:
        a, b = [
            next(r for r in general if r["vocab"] == v and r["baseline_size_mib"] == s)
            for s in (64, 2048)
        ]
        lines.append(
            f"| {v} | {a['bytes_per_token']:.4f} | {a['token_change_percent']:.2f}% | {b['token_change_percent']:.2f}% |"
        )
    lines += [
        "",
        "## Tool-format breakdown: focused 32K vs published 2 GiB / 32K",
        "",
        "Long code-bearing tool results dominate aggregate bytes. This separate breakdown checks the smaller calls and schemas as well.",
        "",
        "| Format | Examples | Original mean tokens | Focused mean tokens | Token savings |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in tool_kinds:
        old = next(
            r for r in tool_breakdown if r["name"] == "original-2048mib-32768" and r["kind"] == kind
        )
        new = next(
            r for r in tool_breakdown if r["name"] == "focused-64mib-32768" and r["kind"] == kind
        )
        saving = 100 * (1 - new["tokens"] / old["tokens"])
        lines.append(
            f"| {kind} | {old['examples']} | {old['mean_tokens_per_example']:.2f} | {new['mean_tokens_per_example']:.2f} | {saving:.2f}% |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "At 32K, the focused fit saves 14.70% of tokens on tool fixtures versus the matched 64 MiB original, but costs 13.07% extra tokens on the general mixture. Against the published 2 GiB / 32K default, tool-fixture savings are 13.97% and the general-mixture cost is 13.99%. A structured-output slice in a balanced mixture is a more promising next experiment than wholesale replacement with this code-only mixture. This is a controlled specialization experiment. Better encoding of generated tool formats does not demonstrate valid tool selection, schema adherence, applicable generated patches or functional code. Repeated shared schema/template structure can favor the focused candidates. Confirm findings with independent real tool traces before promoting a tokenizer or changing the model-training mixture.",
        "",
        "The original 32K tokenizer remains the default. Treat the new fits as research candidates; their general-text regressions and practical model-level results must inform any replacement.",
        "",
        "## Evidence",
        "",
        "`measurements.json` includes byte/token counts, mean and p95 tokens per example, exact round-trip/UNK checks and fixed-fragment token counts. `paired-comparisons.json` retains both matched-64-MiB and original-2-GiB contrasts with intervals. Corpus, source files and tokenizers are hashed. Full per-parent measurements and new tokenizer artifacts remain on Modal under `tokenizer-research/structured-code-v1/`. The local report regenerates from `runs/structured-code-v1/scores.json`.",
        "",
        "Run: `modal run --detach modal_structured_tokenizer.py`, then `python structured_tokenizer_report.py`. A fresh corpus version is required for a rerun; existing artifact paths refuse overwrite. Resource bounds and actual test outcomes are in `execution.json`.",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    with PdfPages(ROOT / "research-report.pdf") as pdf:
        import textwrap

        cover = plt.figure(figsize=(8.27, 11.69))
        cover.text(0.09, 0.92, "Structured-output tokenizer study", fontsize=20, weight="bold")
        cover.text(0.09, 0.87, "21 candidates • Modal CPU • 2026-10-16", fontsize=12)
        highlights = [
            "18 existing tokenizers plus three fresh 64 MiB code-focused fits. Four workloads: Python excerpts, JSON, Git diffs and tool calls. 512 held-out source parents per workload; all candidates pass exact round-trip and zero-UNK checks.",
            "Focused fitting mixture: 40% Python, 20% JSON, 20% diffs and 20% tool-call formats by bytes. Same BPE rules and vocabulary sizes (16K, 32K, 50K).",
            "32K focused vs original 64 MiB: "
            + "; ".join(
                f"{c}: {next(r['token_saving_percent'] for r in contrasts if r['vocab'] == 32768 and r['category'] == c and r['baseline_size_mib'] == 64):.2f}% fewer tokens"
                for c in CATEGORIES
            )
            + ".",
            f"On the original scholarly mixture, the focused 32K tokenizer uses {next(r['token_change_percent'] for r in general if r['vocab'] == 32768 and r['baseline_size_mib'] == 64):.2f}% more tokens than the original 64 MiB / 32K fit.",
            "Caution: JSON, patches and tool calls are generated fixtures, with shared templates across disjoint source parents. This tests encoding efficiency, not production tool-call quality or model correctness. Existing published default is unchanged.",
            "Full tables, methods and limitations are in REPORT.md; machine-readable results, paired intervals and artifact hashes accompany this PDF.",
        ]
        y = 0.80
        for paragraph in highlights:
            wrapped = textwrap.fill(paragraph, 80)
            cover.text(0.09, y, wrapped, fontsize=10, va="top", linespacing=1.4)
            y -= 0.023 * (wrapped.count("\n") + 1) + 0.04
        pdf.savefig(cover)
        plt.close(cover)
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    generate()
