"""Context and output-budget sensitivity, with every checkpoint held fixed."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .investigate import file_sha, request_seed
from .study import command_metrics


def run(variant, experiment, deadline):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = Path("/runs/agentic-reference", variant["model_run"], "model")
    before = file_sha(model / "model.safetensors")
    assert before == variant["sha256"]
    examples = json.loads(Path("/probe-prompts.json").read_text())
    if variant["name"] != "original":
        examples = [e for e in examples if e["view"] == "full"]
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
    out = Path("/runs/weight-space", experiment, "probes", variant["name"])
    out.mkdir(parents=True, exist_ok=False)
    requests = []
    for example in examples:
        for seed in (11, 29, 47):
            for budget in (1536, 4096) if example["group"] == "truncation" else (1536,):
                requests.append({**example, "run_seed": seed, "budget": budget})
    all_rows = []
    for begin in range(0, len(requests), 48):
        if time.time() > deadline - 120:
            raise TimeoutError("Frozen-probe deadline")
        batch = requests[begin : begin + 48]
        prompts = [
            {"prompt_token_ids": tokenizer.encode(r["prompt"], add_special_tokens=False)}
            for r in batch
        ]
        params = [
            SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_tokens=r["budget"],
                stop_token_ids=[151645, 151643],
                skip_special_tokens=False,
                seed=request_seed(r["run_seed"], r["id"], r["turn"]),
            )
            for r in batch
        ]
        outputs = engine.generate(prompts, params, use_tqdm=False)
        rows = []
        for r, output in zip(batch, outputs, strict=True):
            c = output.outputs[0]
            tokens = list(c.token_ids)
            grams = [tuple(tokens[i : i + 8]) for i in range(max(0, len(tokens) - 7))]
            segments = c.text.split(" | ")
            rows.append(
                {k: v for k, v in r.items() if k not in ("prompt", "seen_commands")}
                | {
                    "text": c.text,
                    "token_ids": tokens,
                    "tokens": len(tokens),
                    "finish_reason": c.finish_reason,
                    "stop_reason": c.stop_reason,
                    "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                    "repeat_8gram_fraction": 1 - len(set(grams)) / max(1, len(grams)),
                    "duplicate_pipeline_fragments": sum(
                        n - 1 for s, n in Counter(segments).items() if len(s) > 10
                    ),
                    **command_metrics(c.text, r["seen_commands"]),
                }
            )
        all_rows.extend(rows)
        (out / f"batch-{begin:04d}.json").write_text(json.dumps(rows, indent=2))
        print(
            json.dumps(
                {"variant": variant["name"], "completed": len(all_rows), "total": len(requests)}
            ),
            flush=True,
        )
    assert file_sha(model / "model.safetensors") == before
    (out / "results.json").write_text(json.dumps(all_rows, indent=2))
    return {"variant": variant["name"], "rows": len(all_rows), "weights_unchanged": True}
