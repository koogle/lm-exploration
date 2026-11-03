"""Summarize fixed-context probes with paired case-level uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit_episodes import SEARCH, WRITE, duplicate_segments


def totals(rows):
    return {
        "n": len(rows),
        **{
            key: sum(r[key] for r in rows)
            for key in (
                "valid",
                "repeated",
                "near_repeat",
                "internal_repeat",
                "valid_distinct",
                "valid_distinct_including_pipes",
            )
        },
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "search_or_read_markers": sum(bool(SEARCH.search(r["command"] or "")) for r in rows),
        "write_markers": sum(bool(WRITE.search(r["command"] or "")) for r in rows),
        "mean_tokens": float(np.mean([r["tokens"] for r in rows])),
        "mean_repeat_8gram_fraction": float(np.mean([r["repeat_8gram_fraction"] for r in rows])),
    }


def compare(left, right):
    a = {(r["id"], r["run_seed"]): r for r in left}
    b = {(r["id"], r["run_seed"]): r for r in right}
    assert a.keys() == b.keys()
    cases = sorted({key[0] for key in a})
    rng = np.random.default_rng(20261103)
    result = {}
    for metric in (
        "valid_distinct",
        "valid_distinct_including_pipes",
        "repeated",
        "near_repeat",
        "valid",
    ):
        diffs = np.array(
            [
                np.mean([int(b[k][metric]) - int(a[k][metric]) for k in a if k[0] == case])
                for case in cases
            ]
        )
        boot = rng.choice(diffs, size=(10000, len(cases)), replace=True).mean(1)
        result[metric] = {
            "difference": float(diffs.mean()),
            "case_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    models = {}
    report = {"models": {}, "context_paired": {}, "budget": {}}
    for path in sorted((root / "probes").glob("*.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            # Retain the original score and add the stricter pipeline-aware
            # diagnostic. Raw outputs and pre-existing study scores stay intact.
            row["valid_distinct_including_pipes"] = bool(
                row["valid_distinct"] and not duplicate_segments(row["command"] or "")
            )
        models[path.stem] = rows
        groups = defaultdict(list)
        for row in rows:
            groups[f"{row['group']}/{row['view']}/{row['budget']}"].append(row)
        report["models"][path.stem] = {key: totals(values) for key, values in groups.items()}
        short = {
            (r["id"], r["run_seed"]): r
            for r in rows
            if r["group"] == "truncation" and r["budget"] == 1536
        }
        long = {
            (r["id"], r["run_seed"]): r
            for r in rows
            if r["group"] == "truncation" and r["budget"] == 4096
        }
        budget = []
        for key, a in short.items():
            b = long[key]
            budget.append(
                {
                    "id": key[0],
                    "seed": key[1],
                    "short_length": a["finish_reason"] == "length",
                    "prefix_equal": b["token_ids"][: len(a["token_ids"])] == a["token_ids"],
                    "long_valid": b["valid"],
                    "long_length": b["finish_reason"] == "length",
                    "long_tokens": b["tokens"],
                    "long_repeat_8gram": b["repeat_8gram_fraction"],
                }
            )
        report["budget"][path.stem] = {
            "cases": budget,
            "short_truncated": sum(r["short_length"] for r in budget),
            "prefix_matched_truncated": sum(
                r["short_length"] and r["prefix_equal"] for r in budget
            ),
            "recovered_with_matching_prefix": sum(
                r["short_length"] and r["prefix_equal"] and r["long_valid"] for r in budget
            ),
            "long_truncated": sum(r["long_length"] for r in budget),
        }
    if "original" in models:
        full = [r for r in models["original"] if r["group"] == "loop" and r["view"] == "full"]
        for view in ("recent4", "clip_outputs", "dedup"):
            report["context_paired"][view] = compare(
                full, [r for r in models["original"] if r["group"] == "loop" and r["view"] == view]
            )
        report["model_paired"] = {
            name: compare(full, [r for r in rows if r["group"] == "loop"])
            for name, rows in models.items()
            if name != "original"
        }
    (root / "probe-analysis.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"models": report["models"], "context_paired": report["context_paired"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
