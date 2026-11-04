"""Episode-level looping and action diagnostics, preserving task/seed pairing."""

import argparse
import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path

import numpy as np

WRITE = re.compile(
    r"write_text\(|write_bytes\(|\.write\(|apply_patch|sed\s+-i|perl\s+-[a-z]*i|(?:cat|echo|printf)[^\n]*>[^>]|\btee\s"
)
TEST = re.compile(r"\b(?:pytest|unittest)\b|\bgo test\b|\b(?:npm|yarn|pnpm) (?:run )?test\b")
SEARCH = re.compile(r"\b(?:grep|rg|find|ls|cat|head|tail|sed|nl)\b")


def duplicate_segments(command):
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        pieces = []
        piece = []
        for token in lexer:
            if token in (";", "&&", "||", "|"):
                if len(piece) >= 3:
                    pieces.append(tuple(piece))
                piece = []
            else:
                piece.append(token)
        if len(piece) >= 3:
            pieces.append(tuple(piece))
        return len(pieces) - len(set(pieces))
    except ValueError:
        return None


def episode(row, seed=None):
    seen = {}
    counts = Counter()
    actions = []
    write_epoch = 0
    turn = 0
    first_write = None
    streak = 0
    longest = 0
    for pos, message in enumerate(row["messages"]):
        if message["role"] != "assistant":
            continue
        turn += 1
        calls = message.get("tool_calls", [])
        results = row["messages"][pos + 1 : pos + 1 + len(calls)]
        for index, call in enumerate(calls):
            if call["function"]["name"] != "bash":
                continue
            command = json.loads(call["function"]["arguments"]).get("command", "")
            result = (
                results[index].get("content", "")
                if index < len(results) and results[index]["role"] == "tool"
                else ""
            )
            result_sha = hashlib.sha256(result.encode()).hexdigest()
            write = bool(WRITE.search(command)) and not re.search(
                r"(?:git diff|cat)[^\n]*>\s*patch\.txt", command
            )
            if write:
                write_epoch += 1
                if first_write is None:
                    first_write = turn
            previous = seen.get(command)
            repeated = previous is not None
            stale = repeated and previous == (result_sha, write_epoch)
            seen[command] = (result_sha, write_epoch)
            streak = streak + 1 if stale else 0
            longest = max(longest, streak)
            duplicate = duplicate_segments(command)
            code = re.match(r"exit code: (-?\d+)\n", result)
            flags = {
                "calls": 1,
                "exact_repeats": repeated,
                "same_result_repeats_without_detected_write": stale,
                "possible_writes": write,
                "test_commands": bool(TEST.search(command)),
                "search_or_read_commands": bool(SEARCH.search(command)),
                "duplicate_shell_segments": bool(duplicate),
                "nonzero_exit": bool(code and int(code.group(1)) != 0),
                "tool_timeout": result.startswith("exit code: timeout\n"),
                "missing_path": bool(re.search(r"No such file or directory|cannot access", result)),
                "missing_tool": bool(re.search(r"command not found|not found: command", result)),
                "shell_syntax_error": "syntax error near unexpected token" in result,
                "empty_output": result.strip() == "exit code: 0",
            }
            counts.update({key: int(value) for key, value in flags.items()})
            actions.append(
                {
                    "turn": turn,
                    **flags,
                    "command_chars": len(command),
                    "result_chars": len(result),
                    "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                }
            )
    total = counts["calls"]
    return {
        "id": f"{row['repository']}@{row['sha']}",
        "repository": row["repository"],
        "seed": seed,
        "end": row["end"],
        "resolved": row["resolved"],
        "turns": row["turns"],
        "has_patch": bool(row.get("patch", "").strip()),
        "first_possible_write_turn": first_write,
        "longest_stale_repeat_streak": longest,
        "counts": dict(counts),
        "actions": actions,
        "repeat_fraction": counts["exact_repeats"] / max(1, total),
        "stale_repeat_fraction": counts["same_result_repeats_without_detected_write"]
        / max(1, total),
    }


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row["counts"])
    bins = {}
    for lo, hi in [(1, 10), (11, 20), (21, 40), (41, 60), (61, 100)]:
        actions = [a for r in rows for a in r["actions"] if lo <= a["turn"] <= hi]
        bins[f"{lo}-{hi}"] = {
            "calls": len(actions),
            "exact_repeats": sum(a["exact_repeats"] for a in actions),
            "possible_writes": sum(a["possible_writes"] for a in actions),
        }
    return {
        "episodes": len(rows),
        "resolved": sum(r["resolved"] for r in rows),
        "ends": dict(Counter(r["end"] for r in rows)),
        "counts": dict(counts),
        "episodes_with_patch": sum(r["has_patch"] for r in rows),
        "episodes_with_possible_write": sum(
            r["first_possible_write_turn"] is not None for r in rows
        ),
        "mean_episode_repeat_fraction": float(np.mean([r["repeat_fraction"] for r in rows])),
        "mean_episode_stale_repeat_fraction": float(
            np.mean([r["stale_repeat_fraction"] for r in rows])
        ),
        "stale_streak_quantiles": np.quantile(
            [r["longest_stale_repeat_streak"] for r in rows], [0.5, 0.9, 1]
        ).tolist(),
        "turn_bins": bins,
    }


def paired(
    left,
    right,
    metrics=("resolved", "has_patch", "repeat_fraction", "stale_repeat_fraction", "turns"),
):
    a = {(r["id"], r["seed"]): r for r in left}
    b = {(r["id"], r["seed"]): r for r in right}
    if a.keys() != b.keys():
        raise ValueError("Paired task/seed cohorts must match exactly")
    tasks = sorted({key[0] for key in a})
    result = {}
    rng = np.random.default_rng(20261103)
    for metric in metrics:
        # Resample whole tasks, retaining all available paired seeds per task.
        # A censored seed must not increase its task's point-estimate weight.
        differences = np.array(
            [
                np.sum([float(b[k][metric]) - float(a[k][metric]) for k in a if k[0] == task])
                for task in tasks
            ]
        )
        counts = np.array([sum(k[0] == task for k in a) for task in tasks])
        selected = rng.integers(len(tasks), size=(10000, len(tasks)))
        boot = differences[selected].sum(1) / counts[selected].sum(1)
        result[metric] = {
            "difference": float(differences.sum() / counts.sum()),
            "task_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "degenerate_empirical_bootstrap": bool(np.ptp(boot) == 0),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for spec in args.inputs:
        name, path = spec.split("=", 1)
        rows = [episode(json.loads(line)) for line in Path(path).read_text().splitlines()]
        report[name] = summarize(rows)
        (out / f"episodes-{name}.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({name: report[name]}), flush=True)
    (out / "episode-audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
