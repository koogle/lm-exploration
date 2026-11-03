"""Summarize live trajectories and match terminal turns to unparsed generations."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluation.agentic_behavior import behavior
from training_data import open_swe_format as swe
from training_data import qwen_agent_chat as chat
from transformers import AutoTokenizer

from .study import command_metrics


def analyze(trajectories, raw, tokenizer):
    rows = [json.loads(line) for line in Path(trajectories).read_text().splitlines()]
    generations = [json.loads(line) for line in Path(raw).read_text().splitlines()]
    by_prompt = {row["prompt_sha256"]: row for row in generations}
    terminals, episodes = [], []
    for row in rows:
        task_id = f"{row['repository']}@{row['sha']}"
        seen, counts = [], Counter()
        for message in row["messages"]:
            for call in message.get("tool_calls", []):
                if call["function"]["name"] != "bash":
                    continue
                body = json.dumps(
                    {"name": "bash", "arguments": json.loads(call["function"]["arguments"])}
                )
                metrics = command_metrics(f"<tool_call>{body}</tool_call>", seen)
                counts["calls"] += 1
                for key in ("repeated", "near_repeat", "internal_repeat", "valid_distinct"):
                    counts[key] += metrics[key]
                if metrics["command"] is not None:
                    seen.append(metrics["command"])
        episodes.append({"id": task_id, "end": row["end"], **counts})
        if row["end"] == "answered":
            messages = row["messages"][:-1]
            prompt = chat.render(swe.conversation(messages), swe.TOOLS, True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            generation = by_prompt.get(digest)
            if generation is None:
                raise ValueError(f"No raw generation for terminal turn: {task_id}")
            terminals.append({"id": task_id, **generation})
    totals = Counter()
    for row in episodes:
        totals.update({k: v for k, v in row.items() if k not in ("id", "end")})
    return {
        "behavior": behavior(rows),
        "command_counts": dict(totals),
        "raw_generations": len(generations),
        "raw_finish_reasons": dict(Counter(row["finish_reason"] for row in generations)),
        "terminal_finish_reasons": dict(Counter(row["finish_reason"] for row in terminals)),
        "terminal_outputs": terminals,
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.trajectories, args.raw, AutoTokenizer.from_pretrained(args.tokenizer))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("episodes", "terminal_outputs")})
    )


if __name__ == "__main__":
    main()
