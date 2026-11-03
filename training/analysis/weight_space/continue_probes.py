"""Continue captured truncated text, separating search loops from long patch writes."""

import json
import time
from pathlib import Path

from .investigate import file_sha, request_seed
from .study import command_metrics


def run(variant, experiment, deadline):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = [
        r
        for r in json.loads(Path("/continuation.json").read_text())
        if r["variant"] == variant["name"]
    ]
    model = Path("/runs/agentic-reference", variant["model_run"], "model")
    before = file_sha(model / "model.safetensors")
    assert before == variant["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    engine = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=40960,
        hf_overrides={"max_position_embeddings": 40960},
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    rows = []
    prompts = []
    params = []
    metadata = []
    for row in examples:
        for seed in (11, 29, 47):
            ids = tokenizer.encode(row["prompt"] + row["prefix"], add_special_tokens=False)
            assert len(ids) + 2560 <= 40960
            prompts.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=2560,
                    stop_token_ids=[151645, 151643],
                    skip_special_tokens=False,
                    seed=request_seed(seed, row["id"], row["turn"]),
                )
            )
            metadata.append((row, seed))
    if time.time() > deadline - 120:
        raise TimeoutError("Continuation deadline")
    outputs = engine.generate(prompts, params, use_tqdm=False)
    for (row, seed), output in zip(metadata, outputs, strict=True):
        completion = output.outputs[0]
        text = row["prefix"] + completion.text
        rows.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "seed": seed,
                "prefix": row["prefix"],
                "continuation": completion.text,
                "tokens_added": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "grep_exclusion_count": text.count("grep -v"),
                **command_metrics(text, row["seen_commands"]),
            }
        )
    assert file_sha(model / "model.safetensors") == before
    out = Path("/runs/weight-space", experiment, "continuation", variant["name"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    return {
        "variant": variant["name"],
        "n": len(rows),
        "valid_completed_calls": sum(r["valid"] for r in rows),
        "length": sum(r["finish_reason"] == "length" for r in rows),
        "weights_unchanged": True,
    }
