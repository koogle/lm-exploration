"""Matched single-sequence rollout throughput, including target storage."""

import time

from core import atomic_json, summarize, validate
from megakernel.targets import trajectory


def measure(decoder, tokenizer, manifest, out, volume, report, save):
    report["target_throughput_trials"] = []
    for round_id in range(2):
        target_rows = []
        started_trial = time.perf_counter()
        for repeat in range(8):
            for prompt in manifest["prompts"]:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                result = trajectory(
                    decoder,
                    ids,
                    max_tokens=128,
                    temperature=0,
                    eos_ids=[tokenizer.eos_token_id],
                )
                output = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
                row = {
                    "prompt_id": prompt["id"],
                    "prompt_tokens": len(ids),
                    "verification_seconds": 0.0,
                    "output": output,
                    "finish_reason": result["finish_reason"],
                    "output_tokens": len(result["token_ids"]),
                    "error": None,
                    "validation": validate(prompt, output, result["finish_reason"]),
                    "trajectory": result,
                }
                atomic_json(
                    out / f"targets-r{round_id}-{repeat}-{prompt['id']}.json", row
                )
                volume.commit()
                target_rows.append(row)
        summary = summarize(
            target_rows,
            time.perf_counter() - started_trial,
            manifest["config"]["gpu_usd_per_hour"],
        )
        report["target_throughput_trials"].append(summary)
        save()
