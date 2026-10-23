"""vLLM generation for the teacher and the Qwen3.5-0.8B control (one GPU, greedy).

Modes:
- ``tests``: teacher writes hidden asserts for held-out curriculum problems.
- ``solve``: teacher solves eval items through its chat template (the ceiling).
- ``raw``: a base/fine-tuned model completes the training-format prompt.

Usage: python vllm_eval.py MODE MODEL REVISION ITEMS.json OUT.json
"""

import json
import sys
import time
from pathlib import Path

from evalsets import (
    SOLVE_PROMPT,
    TEST_PROMPT,
    parse_code_string,
    parse_json_answer,
    student_prompt,
    valid_tests,
)

STOPS = ['"explanation"', "<|im_end|>", "<|endoftext|>"]


def main(mode, model, revision, items_path, out_path):
    from vllm import LLM, SamplingParams

    items = json.loads(Path(items_path).read_text())
    started = time.time()
    hub = not model.startswith("/")  # local paths are text-only fine-tuned checkpoints
    llm = LLM(
        model=model,
        revision=revision if hub else None,
        tokenizer_revision=revision if hub else None,
        dtype="bfloat16",
        download_dir="/runs/hf-cache",
        language_model_only=hub,
        max_model_len=4096,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    loaded = time.time() - started
    if mode == "raw":
        prompts = ["<|endoftext|>" + student_prompt(item["problem"]) for item in items]
        params = SamplingParams(temperature=0, max_tokens=768, stop=STOPS)
        outputs = llm.generate(prompts, params)
    else:
        template = TEST_PROMPT if mode == "tests" else SOLVE_PROMPT
        tokenizer = llm.get_tokenizer()
        prompts = [
            {
                "prompt_token_ids": tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": template.format(problem=item["problem"]),
                        }
                    ],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            }
            for item in items
        ]
        params = SamplingParams(temperature=0, max_tokens=2048)
        outputs = llm.generate(prompts, params)
    results = {}
    for item, output in zip(items, outputs):
        text = output.outputs[0].text
        entry = {"text": text, "finish_reason": output.outputs[0].finish_reason}
        if mode == "tests":
            entry["tests"] = valid_tests(parse_json_answer(text, "tests"))
        elif mode == "solve":
            code = parse_json_answer(text, "code")
            entry["code"] = code if isinstance(code, str) else None
        else:
            entry["code"] = parse_code_string(text)
        results[item["id"]] = entry
    report = {
        "mode": mode,
        "model": model,
        "revision": revision,
        "load_seconds": loaded,
        "generation_seconds": time.time() - started - loaded,
        "results": results,
    }
    with open(out_path, "w") as handle:
        json.dump(report, handle)


if __name__ == "__main__":
    main(*sys.argv[1:6])
