"""Generate fresh verified Python/math examples on one H100 with Qwen3-8B."""

import json
import re
from pathlib import Path

import modal

from distillation.modal_vllm import image as base_image
from distillation.modal_vllm import volume

app = modal.App("lm-python-curriculum", include_source=False)
image = base_image
TEACHER = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=7200,
    retries=0,
    volumes={"/distill": volume},
    max_containers=1,
)
def generate(run_id, count):
    import hashlib
    import os
    import shutil
    import time
    from collections import Counter

    os.environ["HF_HOME"] = "/distill/hf-cache"
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distillation.curriculum import task, training_text, verify

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id) or not 32 <= count <= 4096:
        raise ValueError("Expected unique run ID and 32..4096 pilot examples")
    started = time.monotonic()
    output = Path("/distill") / run_id
    output.mkdir(exist_ok=False)
    shutil.copytree(
        "/project/distillation",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__", "results"),
    )
    settings = {
        "teacher": TEACHER,
        "revision": REVISION,
        "requested_examples": count,
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
        "thinking": False,
        "student_tokenizer_sha256": TOKENIZER_SHA256,
    }
    (output / "request.json").write_text(json.dumps(settings, indent=2))
    tasks = [task(i) for i in range(count)]
    if len({t["id"] for t in tasks}) != count:
        raise ValueError("Task specifications repeat; expand curriculum before scaling")
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    volume.commit()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    frozen = load_frozen()
    llm = LLM(
        model=TEACHER,
        revision=REVISION,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        seed=42,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=42)
    instruction = (
        'Return only a JSON object with string keys "code" and "explanation". '
        "Code must define exactly one function named solve, with no imports, decorators, type annotations, helper functions or top-level execution. "
        "Use basic Python loops, arithmetic, indexing and builtins only. Available builtins: abs, all, any, bool, dict, enumerate, int, len, list, max, min, range, reversed, set, sorted, sum, tuple, zip. "
        "Allowed methods: append, add, get, count, index, reverse, sort, copy. Do not mutate inputs. "
        "Explain the mathematical meaning and why the algorithm works in 2-4 sentences, including a small worked example and an edge case. "
        "Keep code and explanation concise. No markdown fences around JSON."
    )
    counters = Counter()
    hashes = set()
    generation_started = time.monotonic()
    for offset in range(0, count, 32):
        batch = tasks[offset : offset + 32]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": t["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for t in batch
        ]
        results = llm.generate(prompts, params, use_tqdm=False)
        for t, result in zip(batch, results, strict=True):
            completion = result.outputs[0]
            record = {k: v for k, v in t.items() if k != "cases"}
            record.update(
                raw=completion.text,
                teacher_tokens=len(completion.token_ids),
                finish_reason=completion.finish_reason,
            )
            counters["attempted"] += 1
            counters["teacher_generated_tokens"] += len(completion.token_ids)
            try:
                answer = json.loads(completion.text)
                if (
                    completion.finish_reason != "stop"
                    or not isinstance(answer, dict)
                    or not all(
                        isinstance(answer.get(key), str) and answer[key].strip()
                        for key in ("code", "explanation")
                    )
                ):
                    raise ValueError("Incomplete or invalid answer")
                verification = verify(answer["code"], t["cases"])
                record.update(
                    answer=answer,
                    verification=verification,
                    explanation_status="unverified_teacher_prose",
                )
                if verification["passed"]:
                    text = training_text(record)
                    fingerprint = hashlib.sha256(text.encode()).hexdigest()
                    if fingerprint in hashes:
                        raise ValueError("Duplicate record")
                    hashes.add(fingerprint)
                    record.update(
                        text_sha256=fingerprint,
                        student_tokens=len(frozen.encode(text).ids) + 1,
                    )
                    counters["accepted"] += 1
                    counters[t["split"] + "_records"] += 1
                    counters[t["split"] + "_student_tokens"] += record["student_tokens"]
                    with (output / "accepted.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                else:
                    counters["rejected_verification"] += 1
            except (ValueError, TypeError, KeyError):
                record["verification"] = {
                    "passed": False,
                    "reason": "invalid_response_or_duplicate",
                }
                counters["rejected_format"] += 1
            with (output / "rollouts.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        elapsed = time.monotonic() - generation_started
        progress = dict(
            counters,
            generation_seconds=elapsed,
            accepted_student_tokens_per_second=(
                counters["train_student_tokens"] + counters["validation_student_tokens"]
            )
            / elapsed,
        )
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
        volume.commit()
    report = dict(
        settings=settings,
        **progress,
        total_seconds=time.monotonic() - started,
        corpus_sha256=hashlib.sha256(
            (output / "accepted.jsonl").read_bytes()
        ).hexdigest()
        if counters["accepted"]
        else None,
        status="completed",
        scope="bounded template pilot; not a 500M-token corpus or proof of scaling",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(run_id: str, count: int = 512):
    report = generate.remote(run_id, count)
    Path("distillation/results").mkdir(exist_ok=True)
    Path(f"distillation/results/{run_id}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
