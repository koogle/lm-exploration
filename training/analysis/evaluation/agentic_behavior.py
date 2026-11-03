"""How an agent behaved in PR-task episodes, from an evaluation's `trajectories.jsonl`.

Run from training/ on downloaded evaluation outputs (`modal_scripts/modal_agentic.py`):

    python -m evaluation.agentic_behavior <trajectories.jsonl> [...]
    python -m evaluation.agentic_behavior --mix <shard.parquet> [...]

Prints one JSON object per file: episodes, resolved, how episodes ended, assistant
turns and the share that carry a tool call, calls per tool, the share of calls that
repeat an earlier call of the same episode (same tool and arguments), edits attempted
and applied (`edit_file`/`write_file` results), and episodes that changed the
repository (a non-empty patch) or submitted. `--mix` reads training conversations
published by `modal_agent_sft.py::publish_mix` (parquet with `source` and JSON
`messages`) and prints, per source, the same share of repeated calls and the
trajectories in which more than 30% of calls repeat.
"""

import collections
import json
import sys
from pathlib import Path

from training_data import pr_task_agent as agent


def repeats(messages):
    """(tool calls, calls repeating an earlier call of the episode: same tool and arguments)."""
    seen, calls, repeated = set(), 0, 0
    for message in messages:
        for call in message.get("tool_calls") or [] if message["role"] == "assistant" else []:
            key = (call["function"]["name"], json.dumps(agent.arguments(call), sort_keys=True))
            calls += 1
            repeated += key in seen
            seen.add(key)
    return calls, repeated


def mix_repeats(paths):
    """Per source of published training conversations: trajectories, tool calls, the
    share of repeated calls and the trajectories with more than 30% repeated calls."""
    from pyarrow import parquet

    totals = collections.defaultdict(lambda: [0, 0, 0, 0])
    for path in paths:
        for row in parquet.read_table(path, columns=["source", "messages"]).to_pylist():
            calls, repeated = repeats(json.loads(row["messages"]))
            total = totals[row["source"]]
            total[0] += 1
            total[1] += calls
            total[2] += repeated
            total[3] += bool(calls) and repeated / calls > 0.3
    return {
        source: {
            "trajectories": n,
            "tool_calls": calls,
            "repeated_calls": round(repeated / calls, 4) if calls else None,
            "trajectories_over_30_percent": loops,
        }
        for source, (n, calls, repeated, loops) in sorted(totals.items())
    }


def behavior(rows):
    ends, tools = collections.Counter(), collections.Counter()
    turns = with_call = calls = repeats = edits = applied = patched = submitted = 0
    for row in rows:
        ends[row["end"]] += 1
        patched += bool(row.get("patch", "").strip())
        submitted += row["end"] == "submitted"
        seen = set()
        messages = row["messages"]
        for position, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            turns += 1
            message_calls = message.get("tool_calls") or []
            # Results follow their turn in call order (call ids need not be unique).
            following = messages[position + 1 : position + 1 + len(message_calls)]
            results = [m.get("content", "") for m in following if m["role"] == "tool"]
            with_call += bool(message_calls)
            for index, call in enumerate(message_calls):
                name = call["function"]["name"]
                key = (name, json.dumps(agent.arguments(call), sort_keys=True))
                calls += 1
                tools[name] += 1
                repeats += key in seen
                seen.add(key)
                if name in ("edit_file", "write_file"):
                    edits += 1
                    result = results[index] if index < len(results) else ""
                    applied += result.startswith(("edited", "wrote"))
    share = lambda part, whole: round(part / whole, 4) if whole else None
    return {
        "episodes": len(rows),
        "resolved": sum(r["resolved"] for r in rows),
        "ends": dict(sorted(ends.items())),
        "assistant_turns": turns,
        "turns_with_tool_call": share(with_call, turns),
        "tool_calls": calls,
        "calls_by_tool": {k: share(v, calls) for k, v in tools.most_common()},
        "repeated_calls": share(repeats, calls),
        "edits": {"attempted": edits, "applied": applied},
        "episodes_with_patch": patched,
        "episodes_submitted": submitted,
    }


def main():
    if sys.argv[1:2] == ["--mix"]:
        print(json.dumps(mix_repeats(sys.argv[2:])))
        return
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        print(json.dumps({"file": path, **behavior(rows)}))


if __name__ == "__main__":
    main()
