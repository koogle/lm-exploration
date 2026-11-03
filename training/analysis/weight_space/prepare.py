"""Build matched novel/repeated command contexts from an explicit evaluation archive."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


def source_module(name):
    spec = importlib.util.spec_from_file_location(name, Path("data") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    qwen = source_module("qwen_agent_chat")
    swe = source_module("open_swe_format")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = Path(args.trajectories).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    examples = []
    for row in rows:
        seen = {}
        for position, message in enumerate(row["messages"]):
            calls = message.get("tool_calls", [])
            if len(calls) != 1 or calls[0]["function"]["name"] != "bash":
                continue
            command = json.loads(calls[0]["function"]["arguments"]).get("command", "")
            if command not in seen:
                seen[command] = position
                continue
            if position < 8 or not 5 < len(tokenizer.encode(command)) < 150:
                continue
            prompt = qwen.render(swe.conversation(row["messages"][:position]), swe.TOOLS, True)
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if length > 16000:
                break  # Later contexts can only grow.
            novel = qwen.render(swe.conversation(row["messages"][: seen[command]]), swe.TOOLS, True)
            examples.append(
                {
                    "id": f"{row['repository']}@{row['sha']}",
                    "repository": row["repository"],
                    "sha": row["sha"],
                    "turn": position // 2,
                    "prompt": prompt,
                    "novel_prompt": novel,
                    "target": qwen._assistant_body(message),
                    "command": command,
                    "seen_commands": list(seen),
                    "prompt_tokens": length,
                }
            )
            break
    examples.sort(key=lambda e: hashlib.sha256(e["id"].encode()).hexdigest())
    if len(examples) <= 24 or len({e["id"] for e in examples}) != len(examples):
        raise ValueError("Need at least 25 distinct tasks for disjoint calibration and test sets")
    for index, example in enumerate(examples):
        example["split"] = "calibration" if index < 24 else "test"
    Path(args.output).write_text(json.dumps(examples, indent=2) + "\n")
    Path(args.output).with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "source": args.trajectories,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "calibration_tasks": 24,
                "test_tasks": len(examples) - 24,
                "selection": "first repeated command after at least three assistant turns; context <=16000 tokens",
                "partition": "sort task IDs by SHA256, first 24 calibration, remaining test",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"tasks": len(examples), "calibration": 24, "test": len(examples) - 24}))


if __name__ == "__main__":
    main()
