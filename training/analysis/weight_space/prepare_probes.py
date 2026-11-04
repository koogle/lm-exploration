"""Build late-loop context ablations and known truncated-reply controls."""

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from training_data import open_swe_format as swe
from training_data import qwen_agent_chat as chat
from transformers import AutoTokenizer


def commands(messages):
    return [
        json.loads(c["function"]["arguments"]).get("command", "")
        for m in messages
        for c in m.get("tool_calls", [])
        if c["function"]["name"] == "bash"
    ]


def prompt(messages):
    return chat.render(swe.conversation(messages), swe.TOOLS, True)


def views(messages):
    starts = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    recent = messages[:2] + messages[starts[max(0, len(starts) - 4)] :]
    clipped = copy.deepcopy(messages)
    for message in clipped:
        if message["role"] == "tool" and len(message["content"]) > 1000:
            # Keep the actual return-code header and a visible truncation marker.
            message["content"] = (
                message["content"][:700]
                + "\n[diagnostic output clipping]\n"
                + message["content"][-200:]
            )
    # Remove duplicate complete assistant/result groups, preserving their latest
    # occurrence. This is a context probe, not a live history-management policy.
    groups = [messages[a:b] for a, b in zip(starts, starts[1:] + [len(messages)], strict=True)]
    seen, kept = set(), []
    for group in reversed(groups):
        # Call ids are arbitrary harness bookkeeping and do not enter the prompt.
        rendered = prompt(messages[:2] + group)
        key = hashlib.sha256(rendered.encode()).hexdigest()
        if key not in seen:
            kept.append(group)
            seen.add(key)
    dedup = messages[:2] + [m for group in reversed(kept) for m in group]
    return {"full": messages, "recent4": recent, "clip_outputs": clipped, "dedup": dedup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--round2-live", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(s) for s in Path(args.trajectories).read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        seen = Counter()
        turn = 0
        for pos, m in enumerate(row["messages"]):
            if m["role"] != "assistant":
                continue
            turn += 1
            current = commands([m])
            if len(current) == 1 and seen[current[0]] >= 2 and turn >= 12:
                messages = row["messages"][:pos]
                if len(tokenizer.encode(prompt(messages), add_special_tokens=False)) <= 30000:
                    groups[row["repository"]].append((row, messages, turn))
                    break
            seen.update(current)
    for group in groups.values():
        group.sort(key=lambda x: hashlib.sha256(("probe:" + x[0]["sha"]).encode()).hexdigest())
    selected = []
    while len(selected) < 48 and any(groups.values()):
        for repo in sorted(groups):
            if groups[repo] and len(selected) < 48:
                selected.append(groups[repo].pop(0))
    assert len(selected) == 48
    examples = []
    for row, messages, turn in selected:
        task_id = f"{row['repository']}@{row['sha']}"
        for view, history in views(messages).items():
            examples.append(
                {
                    "id": task_id + ":loop",
                    "task_id": task_id,
                    "group": "loop",
                    "view": view,
                    "prompt": prompt(history),
                    "seen_commands": commands(messages),
                    "turn": turn,
                }
            )
    for row, _, _ in selected[:16]:
        task_id = f"{row['repository']}@{row['sha']}"
        examples.append(
            {
                "id": task_id + ":first",
                "task_id": task_id,
                "group": "first_turn",
                "view": "full",
                "prompt": prompt(row["messages"][:2]),
                "seen_commands": [],
                "turn": 1,
            }
        )
    excluded = []
    for folder in sorted(Path(args.round2_live).iterdir()):
        for row in map(json.loads, (folder / "trajectories.jsonl").read_text().splitlines()):
            if row["end"] != "answered":
                continue
            messages = row["messages"][:-1]
            task_id = f"{row['repository']}@{row['sha']}"
            text = prompt(messages)
            if len(tokenizer.encode(text, add_special_tokens=False)) + 4096 > 40960:
                excluded.append(
                    {
                        "id": task_id,
                        "run": folder.name,
                        "reason": "Cannot fit 4096 output tokens in the fixed context",
                    }
                )
                continue
            examples.append(
                {
                    "id": task_id + ":" + folder.name + ":terminal",
                    "task_id": task_id,
                    "group": "truncation",
                    "view": "full",
                    "prompt": text,
                    "seen_commands": commands(messages),
                    "turn": row["turns"],
                }
            )
    for row in examples:
        row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
    output = Path(args.output)
    output.write_text(json.dumps(examples, indent=2))
    output.with_suffix(".excluded.json").write_text(json.dumps(excluded, indent=2))
    print(
        json.dumps(
            {
                "groups": dict(Counter(x["group"] for x in examples)),
                "max_prompt_tokens": max(x["prompt_tokens"] for x in examples),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
