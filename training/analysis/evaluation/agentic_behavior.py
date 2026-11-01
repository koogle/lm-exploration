"""How an agent behaved in PR-task episodes, from an evaluation's `trajectories.jsonl`.

Run from training/ on downloaded evaluation outputs (`modal_scripts/modal_agentic.py`):

    python -m evaluation.agentic_behavior <trajectories.jsonl> [...]

Prints one JSON object per file: episodes, resolved, how episodes ended, assistant
turns and the share that carry a tool call, calls per tool, the share of calls that
repeat an earlier call of the same episode (same tool and arguments), edits attempted
and applied (`edit_file`/`write_file` results), and episodes that changed the
repository (a non-empty patch) or submitted.
"""

import collections
import json
import sys
from pathlib import Path

from training_data import pr_task_agent as agent


def behavior(rows):
    ends, tools = collections.Counter(), collections.Counter()
    turns = with_call = calls = repeats = edits = applied = patched = submitted = 0
    for row in rows:
        ends[row["end"]] += 1
        patched += bool(row.get("patch", "").strip())
        submitted += row["end"] == "submitted"
        results = {
            m.get("tool_call_id"): m.get("content", "")
            for m in row["messages"]
            if m["role"] == "tool"
        }
        seen = set()
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            turns += 1
            message_calls = message.get("tool_calls") or []
            with_call += bool(message_calls)
            for call in message_calls:
                name = call["function"]["name"]
                key = (name, json.dumps(agent.arguments(call), sort_keys=True))
                calls += 1
                tools[name] += 1
                repeats += key in seen
                seen.add(key)
                if name in ("edit_file", "write_file"):
                    edits += 1
                    applied += results.get(call.get("id"), "").startswith(("edited", "wrote"))
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
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        print(json.dumps({"file": path, **behavior(rows)}))


if __name__ == "__main__":
    main()
