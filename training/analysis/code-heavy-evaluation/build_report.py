"""Build public aggregates and a separate private all-samples report."""

import argparse
import json
import statistics
from pathlib import Path

LABELS = {"original_5b": "Original 5B", "eu_parent": "EU parent", "final_code60": "Final 10.879B"}
MODE_NAMES = {
    "greedy": "Greedy",
    "low_temperature_topk": "T=0.2, k=10",
    "sampled_topk": "T=0.8, k=50",
}
CATEGORIES = [
    ("original/code_python", "Python code"),
    ("original/fineweb_edu", "FineWeb-Edu"),
    ("original/finemath_4plus", "FineMath"),
    ("original/pes2o", "peS2o"),
    ("original/pubmed", "PubMed"),
    ("original/arxiv", "arXiv"),
    ("original/pressbooks", "Pressbooks"),
    ("original/libretexts", "LibreTexts"),
    ("original/doab", "DOAB"),
    ("original/gutenberg", "Gutenberg"),
    ("original/pre_1929_books", "Pre-1929 books"),
    ("original/library_of_congress", "Library of Congress"),
    ("ecb", "ECB"),
    ("ema", "EMA"),
]


def summarize(data):
    if not data.get("completed") or len(data["samples"]) != 50:
        raise ValueError("Incomplete generation comparison")
    modes = {}
    for mode in MODE_NAMES:
        samples = [r for r in data["samples"] if r["mode"] == mode]

        def aggregate(rows):
            return {
                "samples": len(rows),
                "mean_repeated_token_4gram_fraction": statistics.mean(
                    r["repeated_token_4gram_fraction"] for r in rows
                ),
                "median_repeated_token_4gram_fraction": statistics.median(
                    r["repeated_token_4gram_fraction"] for r in rows
                ),
                "long_loop_samples": sum(r["long_loop"] for r in rows),
                "eos_samples": sum(r["stop_reason"] == "eos" for r in rows),
                "total_generated_tokens": sum(r["new_tokens"] for r in rows),
            }

        modes[mode] = {
            **aggregate(samples),
            "categories": {
                c: aggregate([r for r in samples if r["category"] == c])
                for c in sorted({r["category"] for r in samples})
            },
        }
    return {
        "identity": data["identity"],
        "modes": modes,
        "max_new_tokens": data["max_new_tokens"],
        "elapsed_seconds": data["elapsed_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    raw = {label: json.loads((a.input / (label + ".json")).read_text()) for label in LABELS}
    summary = {label: summarize(d) for label, d in raw.items()}
    # All comparisons must use identical prompt/mode/seed settings.
    keys = lambda d: [(r["prompt"], r["mode"], r["seed"]) for r in d["samples"]]
    if any(keys(d) != keys(raw["final_code60"]) for d in raw.values()):
        raise ValueError("Unmatched sample settings")
    validation = json.loads((a.input / "validation.json").read_text())
    before = validation["comparisons"]["before"]["categories"]
    eu = validation["comparisons"]["after"]["categories"]
    a.output.mkdir(parents=True, exist_ok=True)
    report = [
        "# 124M - 10.879B: final validation and repetition",
        "",
        (
            "Training completed on October 19, 2026. The final checkpoint is step 66,065, "
            "with 4,999,971,081 additional targets and 10,878,562,619 lifetime targets. "
            "The model archive is private: https://huggingface.co/jfrick/124M-10.879B ."
        ),
        "",
        "## Matched validation",
        "",
        (
            "Lower cross-entropy loss (nats/target) is better. These are the same 7,620,187 "
            "held-out targets from 2,782 source-family selections, scored in FP32 with TF32 off. "
            "Windows are at most 2K; this does not establish full-8K quality."
        ),
        "",
        "| Source | Original 5B | EU parent | Final 10.879B |",
        "|---|---:|---:|---:|",
    ]
    for key, name in CATEGORIES:
        report.append(
            f"| {name} | {before[key]['before_loss']:.3f} | {eu[key]['before_loss']:.3f} | {before[key]['after_loss']:.3f} |"
        )
    tokens = sum(x["tokens"] for x in before.values())
    overall = [
        sum(x[k] * x["tokens"] for x in rows.values()) / tokens
        for rows, k in [(before, "before_loss"), (eu, "before_loss"), (before, "after_loss")]
    ]
    report += [
        "",
        (
            f"Token-weighted loss on this fixed evaluation sample: "
            f"{overall[0]:.4f} → {overall[1]:.4f} → {overall[2]:.4f}. "
            "This is a family-sampled evaluation mix, not the training mixture."
        ),
        "",
        (
            "Coding improves over both previous checkpoints. The other 11 original sources "
            "recover from EU forgetting but remain slightly worse than the original 5B model. "
            "ECB and EMA lose most EU-specific gains, while retaining small gains over the original model."
        ),
        "",
        "## Repetition",
        "",
        (
            "150 matched continuations: ten fixed prompts, three checkpoints, one greedy seed and "
            "two seeds per sampled setting. Cap: 256 new tokens; stop on EOS. All generation uses "
            "FP32 with TF32 off; PAD/BOS/UNK are suppressed, EOS is allowed. No repetition penalty "
            "or n-gram blocking is applied. Prompts cover general text, maths, story, three Python "
            "functions, ECB and EMA. This is a small diagnostic, not a general capability benchmark."
        ),
        "",
        (
            "Each cell is the mean fraction of generated token four-grams that repeat an earlier "
            "four-gram in the same continuation. Lower is less exact repetition."
        ),
        "",
        "| Decoder | Samples/model | Original 5B | EU parent | Final 10.879B |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, name in MODE_NAMES.items():
        values = [summary[l]["modes"][mode] for l in LABELS]
        report.append(
            f"| {name} | {values[0]['samples']} | "
            + " | ".join(f"{v['mean_repeated_token_4gram_fraction']:.1%}" for v in values)
            + " |"
        )
    report += [
        "",
        (
            "Long exact loops (at least 64 tokens covered by at least three consecutive "
            "copies of a token pattern of length 1–64):"
        ),
        "",
        "| Decoder | Original 5B | EU parent | Final 10.879B |",
        "|---|---:|---:|---:|",
    ]
    for mode, name in MODE_NAMES.items():
        report.append(
            f"| {name} | "
            + " | ".join(
                f"{summary[l]['modes'][mode]['long_loop_samples']}/{summary[l]['modes'][mode]['samples']}"
                for l in LABELS
            )
            + " |"
        )
    report += [
        "",
        (
            "Repetition is **not solved**: greedy and very low-temperature decoding remain "
            "strongly repetitive. T=0.8/k=50 is less repetitive in this small matched sample, "
            "and improves versus both parents. Zero flagged long loops does not mean no "
            "repetition: near-repeated paragraphs, renamed functions and semantically repeated "
            "sentences may evade that narrow flag. Ordinary code syntax also repeats naturally. "
            "Fewer repeated four-grams do not establish factual or programming correctness."
        ),
        "",
        (
            "All raw samples remain private. The separate all-samples report includes every "
            "prompt, decoder, seed, continuation and repetition result; examples are not "
            "cherry-picked into the aggregate. Public repository artifacts contain aggregates only."
        ),
        "",
        (
            "Validation uses paired family bootstrap intervals; interval and per-source counts "
            "are in validation-detailed-summary.json. They are conditional on selected families "
            "and do not account for selection of prompts or multiple source comparisons. "
            "Split reconciliation, near-duplicate and contamination audits remain pending."
        ),
    ]
    (a.output / "report.md").write_text("\n".join(report) + "\n")
    (a.output / "generation-summary.json").write_text(
        json.dumps(
            {
                "models": summary,
                "modes": raw["final_code60"]["modes"],
                "prompts": raw["final_code60"]["prompts"],
            },
            indent=2,
        )
        + "\n"
    )
    (a.output / "validation-detailed-summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    samples = [
        "# Private generated samples: 124M - 10.879B",
        "",
        (
            "All 150 matched continuations, including failures. Text is raw model output, "
            "not verified factual or correct code. Generated token IDs are in the adjacent JSON files."
        ),
        "",
    ]
    for label, data in raw.items():
        samples += [f"## {LABELS[label]}", ""]
        for row in data["samples"]:
            samples += [
                f"### {MODE_NAMES[row['mode']]} · seed {row['seed']} · prompt {row['prompt_index']}",
                "",
                f"Prompt: {json.dumps(row['prompt'])}",
                "",
                (
                    f"Tokens: {row['new_tokens']}; stop: {row['stop_reason']}; repeated four-grams: "
                    f"{row['repeated_token_4gram_fraction']:.1%}; long exact loop: {row['long_loop']}."
                ),
                "",
                "````text",
                row["text"],
                "````",
                "",
            ]
    (a.input / "all-samples.md").write_text("\n".join(samples) + "\n")
    print(
        json.dumps(
            {
                "report": str(a.output / "report.md"),
                "private_samples": str(a.input / "all-samples.md"),
                "samples": 150,
            }
        )
    )


if __name__ == "__main__":
    main()
