"""Reuse calibration failures and reserve previously unevaluated tasks for round two."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from training_data import open_swe_format as swe
from training_data import qwen_agent_chat as chat
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--previous-prompts", required=True)
    parser.add_argument("--failure-trajectories", required=True)
    parser.add_argument("--source-trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    previous = json.loads(Path(args.previous_prompts).read_text())
    examples = [{**x, "group": "loops"} for x in previous if x["split"] == "calibration"]
    # Controls exercise useful command issuance, not a penalty for repeating a test.
    examples += [
        {
            **x,
            "id": x["id"] + ":first-use",
            "prompt": x["novel_prompt"],
            "seen_commands": [],
            "group": "clean",
        }
        for x in examples[:8]
    ]
    failure_file = Path(args.failure_trajectories)
    for row in map(json.loads, failure_file.read_text().splitlines()):
        if row["end"] != "answered":
            continue
        messages = row["messages"][:-1]
        prompt = chat.render(swe.conversation(messages), swe.TOOLS, True)
        seen = []
        for message in messages:
            for call in message.get("tool_calls", []):
                if call["function"]["name"] == "bash":
                    seen.append(json.loads(call["function"]["arguments"])["command"])
        examples.append(
            {
                "id": f"{row['repository']}@{row['sha']}:failure",
                "group": "format_regression",
                "prompt": prompt,
                "seen_commands": seen,
            }
        )
    for row in examples:
        row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
        if row["prompt_tokens"] + 1536 > 40960:
            raise ValueError("Regression context exceeds the model's position limit")
    source = Path(args.source_trajectories)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    old_ids = {x["id"] for x in previous}
    groups = defaultdict(list)
    for row in rows:
        key = f"{row['repository']}@{row['sha']}"
        if key not in old_ids:
            groups[row["repository"]].append(key)
    for values in groups.values():
        values.sort(key=lambda x: hashlib.sha256(("round2:" + x).encode()).hexdigest())
    if sum(map(len, groups.values())) < 16:
        raise ValueError("Fewer than 16 unused tasks remain")
    live = []
    while len(live) < 16:
        for repo in sorted(groups):
            if groups[repo] and len(live) < 16:
                live.append(groups[repo].pop(0))
    assert not set(live) & old_ids
    (out / "prompts.json").write_text(json.dumps(examples, indent=2) + "\n")
    (out / "live-tasks.json").write_text(json.dumps(live, indent=2) + "\n")
    manifest = {
        "groups": {
            group: sum(x["group"] == group for x in examples)
            for group in {x["group"] for x in examples}
        },
        "max_prompt_tokens": max(x["prompt_tokens"] for x in examples),
        "live_tasks": live,
        "live_selection": "round robin by repository, SHA256 round2:task within repository, excluding all 118 round-one diagnostic tasks",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256((out / "prompts.json").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
