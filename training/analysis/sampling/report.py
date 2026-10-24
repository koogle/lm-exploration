"""Summarize sampling results: repetition metrics, reproducibility and all completions.

Run from training/:
    uv run python -m sampling.report RESULT.json [...] --output-dir DIR \
        [--reference REFERENCE.json ...] [--group-by mode] [--paired]

Reads launcher results (`models[]`). Writes summary.json, metrics-table.md and
completions.md. Makes no model calls and no quality judgements.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sampling.metrics import sample_metrics
from sampling.recipe import MODE_DEFAULTS

DECODING = tuple(MODE_DEFAULTS)
METRICS = ("token_rep4", "word_rep4", "char_rep16")
PREFIX_METRICS = ("char_rep16_first128", "char_rep16_first256", "word_rep4_first32")


def decoding(sample):
    """Decoding settings recorded on the sample; greedy ignores filters."""
    values = {key: sample[key] for key in DECODING}
    if values["temperature"] == 0:
        values.update(top_k=0, top_p=1.0)
    return values


def load(path):
    """Flat sample rows with run label, decoding settings, prompt group and metrics."""
    rows = []
    for run in json.loads(Path(path).read_text())["models"]:
        for sample in run["samples"]:
            rows.append(
                {
                    "run": run["name"],
                    # Volume checkpoints have no HF revision; their weights hash identifies them.
                    "revision": run.get("revision") or run["checkpoint_sha256"],
                    "prompt": sample["prompt"],
                    "prompt_group": sample["prompt_group"],
                    "mode": sample["mode"],
                    "seed": sample["seed"],
                    **decoding(sample),
                    "generated_tokens": sample["generated_tokens"],
                    "ended_with_eos": sample["ended_with_eos"],
                    "generation_ids": sample["generation_ids"],
                    "completion": sample["completion"],
                    **sample_metrics(sample["completion"], sample["generation_ids"]),
                }
            )
    return rows


def match_key(row):
    return (row["revision"], row["prompt"], row["seed"], *(row[k] for k in DECODING))


def compare(rows, references):
    """Attach the first reference sample with the same revision, prompt, seed and decoding."""
    index = {}
    for reference in references:
        index.setdefault(match_key(reference), reference)
    comparisons = []
    for row in rows:
        reference = index.get(match_key(row))
        if reference is None:
            continue
        ours, theirs = row["generation_ids"], reference["generation_ids"]
        divergence = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
        if divergence is None and len(ours) != len(theirs):
            divergence = min(len(ours), len(theirs))
        row["reference"] = reference
        row["token_identical"] = ours == theirs
        comparisons.append(
            {
                "run": row["run"],
                "mode": row["mode"],
                "reference_mode": reference["mode"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "token_identical": ours == theirs,
                "first_divergent_position": divergence,
                "token_rep4": row["token_rep4"],
                "reference_token_rep4": reference["token_rep4"],
            }
        )
    return comparisons


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def table(rows, fields, paired):
    """One row per run and group; --paired limits prefix metrics to keys eligible in every run."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run"], *(row[f] for f in fields))].append(row)
    runs = list(dict.fromkeys(row["run"] for row in rows))
    common = {}
    for metric in PREFIX_METRICS if paired else ():
        eligible = defaultdict(set)
        for row in rows:
            if row[metric] is not None:
                key = (*(row[f] for f in fields), row["prompt"], row["seed"])
                eligible[row["run"]].add(key)
        common[metric] = set.intersection(*(eligible[run] for run in runs))
    result = []
    for (run, *values), selected in groups.items():
        entry = {
            "run": run,
            **dict(zip(fields, values, strict=True)),
            "n": len(selected),
            "eos": sum(s["ended_with_eos"] for s in selected),
            "mean_generated_tokens": mean(s["generated_tokens"] for s in selected),
        }
        for metric in METRICS:
            entry[metric] = mean(s[metric] for s in selected)
        for metric in PREFIX_METRICS:
            included = [
                s
                for s in selected
                if metric not in common
                or (*(s[f] for f in fields), s["prompt"], s["seed"]) in common[metric]
            ]
            entry[metric] = average(s[metric] for s in included)
            entry[f"{metric}_n"] = sum(s[metric] is not None for s in included)
        matched = [s for s in selected if "reference" in s]
        if matched:
            entry["reference_n"] = len(matched)
            entry["token_identical"] = sum(s["token_identical"] for s in matched)
            entry["matched_token_rep4"] = mean(s["token_rep4"] for s in matched)
            entry["reference_token_rep4"] = mean(s["reference"]["token_rep4"] for s in matched)
        result.append(entry)
    return result


def percent(value, n=None):
    if value is None:
        return "—"
    return f"{value:.2%}" + (f" ({n})" if n is not None else "")


def markdown(fields, entries):
    reference = any("reference_n" in e for e in entries)
    columns = ["Run", *fields, "N", "EOS", "Tokens", "Token4", "Word4", "Char16"]
    columns += ["Char16 first128 (n)", "Char16 first256 (n)", "Word4 first32 (n)"]
    columns += ["Ref N", "Identical", "Matched Token4", "Ref Token4"] if reference else []
    lines = ["| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
    for e in entries:
        cells = [e["run"], *(str(e[f]) for f in fields), e["n"], e["eos"]]
        cells += [f"{e['mean_generated_tokens']:.1f}"]
        cells += [percent(e[m]) for m in METRICS]
        cells += [percent(e[m], e[f"{m}_n"]) for m in PREFIX_METRICS]
        if reference:
            cells += [e.get("reference_n", 0), e.get("token_identical", "—")]
            cells += [percent(e.get("matched_token_rep4")), percent(e.get("reference_token_rep4"))]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def fenced(text):
    fence = "`" * max(3, max(map(len, re.findall("`+", text)), default=0) + 1)
    return [f"{fence}text", text, fence]


def completions(rows):
    lines = ["# All completions", "", "Verbatim, unfiltered outputs.", ""]
    for row in rows:
        lines += [f"## {row['run']} / {row['mode']} / seed {row['seed']}", "", "Prompt:", ""]
        lines += [*fenced(row["prompt"]), "", "Completion:", "", *fenced(row["completion"]), ""]
    return "\n".join(lines)


def file_record(path):
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="Result JSONs to summarize")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        default=[],
        help="Earlier results for token-identity checks; the first matching file wins",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        help="Comma-separated sample fields per table (repeatable); 'decoding' expands to "
        f"{','.join(DECODING)}. Default: mode",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Average prefix metrics only over prompt/seed keys eligible in every run",
    )
    args = parser.parse_args(argv)
    rows = [row for path in args.results for row in load(path)]
    references = [row for path in args.reference for row in load(path)]
    comparisons = compare(rows, references)
    tables = []
    for spec in args.group_by or ["mode"]:
        fields = [
            f for part in spec.split(",") for f in (DECODING if part == "decoding" else [part])
        ]
        tables.append({"group_by": fields, "rows": table(rows, fields, args.paired)})
    summary = {
        "inputs": [file_record(p) for p in args.results],
        "references": [file_record(p) for p in args.reference],
        "paired_prefix_metrics": args.paired,
        "tables": tables,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = "\n\n".join(markdown(t["group_by"], t["rows"]) for t in tables) + "\n"
    (args.output_dir / "metrics-table.md").write_text(text)
    (args.output_dir / "completions.md").write_text(completions(rows))
    print(text)


if __name__ == "__main__":
    main()
