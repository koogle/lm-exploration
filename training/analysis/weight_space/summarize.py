"""Rescore saved generations and compare a frozen pair, emitting structured evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from study import command_metrics, summary


def paired(original, candidate):
    left, right = {x["id"]: x for x in original}, {x["id"]: x for x in candidate}
    if left.keys() != right.keys():
        raise ValueError("Paired comparisons require identical task IDs")
    result = {}
    for key in (
        "repeated",
        "near_repeat",
        "internal_repeat",
        "valid",
        "valid_distinct",
        "truncated",
    ):
        differences = np.array([int(right[k][key]) - int(left[k][key]) for k in left])
        rng = np.random.default_rng(20261103)
        samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(1)
        result[key] = {
            "candidate_minus_original": float(differences.mean()),
            "paired_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "zero_to_one": int((differences == 1).sum()),
            "one_to_zero": int((differences == -1).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--original")
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompts = {x["id"]: x for x in json.loads(Path(args.prompts).read_text())}
    rows, summaries = {}, {}
    for name in args.results:
        path = Path(name)
        values = json.loads(path.read_text())
        for row in values:
            row.update(command_metrics(row["text"], prompts[row["id"]]["seen_commands"]))
        rows[path.stem] = values
        summaries[path.stem] = summary(values)
    result = {"summaries": summaries}
    if args.original or args.candidate:
        result["paired"] = paired(rows[args.original], rows[args.candidate])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
