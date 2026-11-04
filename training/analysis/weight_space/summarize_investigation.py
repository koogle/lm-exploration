"""Aggregate the frozen live comparison, including hidden-test partial progress."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .audit_episodes import episode, paired, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    config = json.loads((root / "live-config.json").read_text())
    tasks = {
        r["repository"] + "@" + r["sha"]: r
        for r in map(json.loads, (root / "verified.jsonl").read_text().splitlines())
    }
    expected = {(task, seed) for task in config["task_ids"] for seed in config["seeds"]}
    censor = (
        json.loads((root / "censored.json").read_text())
        if (root / "censored.json").exists()
        else None
    )
    excluded = {(censor["id"], censor["seed"])} if censor else set()
    models = {}
    report = {
        "complete": args.complete,
        "models": {},
        "paired": {},
        "paired_exclusions": [censor] if censor else [],
        "matched_models": {},
    }
    for variant in config["variants"]:
        name = variant["name"]
        rows = []
        raw_counts = Counter()
        token_counts = Counter()
        stage_counts = Counter()
        errors = []
        cases = []
        for folder in sorted((root / "live" / name).glob("seed*-batch*")):
            if not (folder / "raw.jsonl").exists():
                continue
            seed = int(folder.name.split("-")[0][4:])
            raw = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            by_turn = {(r["id"], r["turn"]): r for r in raw}
            scored_raw = [
                r
                for r in raw
                if not (censor and name == censor["variant"] and (r["id"], seed) in excluded)
            ]
            raw_counts.update(r["finish_reason"] for r in scored_raw)
            token_counts.update(
                {
                    "all": sum(r["tokens"] for r in scored_raw),
                    "truncated": sum(
                        r["tokens"] for r in scored_raw if r["finish_reason"] == "length"
                    ),
                }
            )
            tokens_by_task = Counter()
            for r in raw:
                tokens_by_task[r["id"]] += r["tokens"]
            for item in map(json.loads, (folder / "trajectories.jsonl").read_text().splitlines()):
                row = episode(item, seed)
                key = row["id"]
                row["generated_tokens"] = tokens_by_task[key]
                row["censored"] = bool(
                    censor and name == censor["variant"] and (key, seed) in excluded
                )
                if row["censored"]:
                    rows.append(row)
                    continue
                v = tasks[key]["verification"]
                row["patch_paths"] = re.findall(
                    r"^diff --git a/.*? b/(.*?)$", item["patch"], re.MULTILINE
                )
                row["reference_file_overlap"] = sorted(
                    set(row["patch_paths"]) & set(tasks[key]["source_paths"])
                )
                results = item["grade"].get("results", {})
                stage_counts[item["grade"].get("stage")] += 1
                if item["error"]:
                    errors.append({"id": key, "seed": seed, "error": item["error"]})
                f2p = v["fail_to_pass"]
                p2p = v["pass_to_pass"]
                row["f2p_passed"] = sum(results.get(t) == "pass" for t in f2p)
                row["f2p_total"] = len(f2p)
                row["f2p_pass_fraction"] = row["f2p_passed"] / max(1, len(f2p))
                row["p2p_nonpass"] = sum(results.get(t) != "pass" for t in p2p)
                row["answered_truncated"] = False
                if row["end"] == "answered":
                    terminal = by_turn[(key, row["turns"])]
                    row["answered_truncated"] = terminal["finish_reason"] == "length"
                if row["resolved"] or row["f2p_passed"]:
                    cases.append({k: val for k, val in row.items() if k != "actions"})
                rows.append(row)
        keys = {(r["id"], r["seed"]) for r in rows}
        if len(keys) != len(rows):
            raise ValueError("Duplicate task-seed episodes")
        if args.complete and keys != expected:
            raise ValueError(f"Incomplete {name}: {len(keys)}/{len(expected)}")
        all_rows = rows
        rows = [r for r in all_rows if not r["censored"]]
        models[name] = rows
        if not rows:
            continue
        report["models"][name] = {
            **summarize(rows),
            "attempted_episodes": len(all_rows),
            "censored_episodes": sum(r["censored"] for r in all_rows),
            "raw_finish_reasons": dict(raw_counts),
            "generated_tokens": dict(token_counts),
            "grade_stages": dict(stage_counts),
            "errors": errors,
            "episodes_with_f2p_progress": sum(r["f2p_passed"] > 0 for r in rows),
            "episodes_editing_reference_source_files": sum(
                bool(r["reference_file_overlap"]) for r in rows
            ),
            "mean_f2p_pass_fraction": float(np.mean([r["f2p_pass_fraction"] for r in rows])),
            "episodes_with_p2p_nonpass": sum(r["p2p_nonpass"] > 0 for r in rows),
            "truncated_endings": sum(r["answered_truncated"] for r in rows),
            "partial_or_solved_cases": cases,
            "by_seed": {
                str(seed): summarize([r for r in rows if r["seed"] == seed])
                for seed in config["seeds"]
                if any(r["seed"] == seed for r in rows)
            },
        }
        (root / f"episodes-{name}.json").write_text(json.dumps(all_rows, indent=2))
    if args.complete:
        matched = {
            name: [r for r in rows if (r["id"], r["seed"]) not in excluded]
            for name, rows in models.items()
        }
        for name, rows in matched.items():
            assert {(r["id"], r["seed"]) for r in rows} == expected - excluded
            report["matched_models"][name] = {
                **summarize(rows),
                "truncated_endings": sum(r["answered_truncated"] for r in rows),
                "episodes_with_f2p_progress": sum(r["f2p_passed"] > 0 for r in rows),
                "episodes_editing_reference_source_files": sum(
                    bool(r["reference_file_overlap"]) for r in rows
                ),
                "mean_f2p_pass_fraction": float(np.mean([r["f2p_pass_fraction"] for r in rows])),
                "episodes_with_p2p_nonpass": sum(r["p2p_nonpass"] > 0 for r in rows),
                "generated_tokens": sum(r["generated_tokens"] for r in rows),
            }
            if name != "original":
                report["paired"][name] = paired(
                    matched["original"],
                    rows,
                    metrics=(
                        "resolved",
                        "has_patch",
                        "repeat_fraction",
                        "stale_repeat_fraction",
                        "turns",
                        "answered_truncated",
                        "f2p_pass_fraction",
                        "generated_tokens",
                    ),
                )
    (root / "live-analysis.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                name: {
                    k: r[k]
                    for k in [
                        "episodes",
                        "resolved",
                        "ends",
                        "episodes_with_patch",
                        "truncated_endings",
                        "mean_episode_repeat_fraction",
                        "episodes_with_f2p_progress",
                    ]
                }
                for name, r in report["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
