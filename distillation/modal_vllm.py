"""vLLM jobs on Modal: curriculum generation and export parity. Importing starts no job.

Run from the repository root:
    modal run --detach -m distillation.modal_vllm --recipe <curriculum.json> \
        [--run-id <fresh-id>] [--output <result.json>]
    modal run -m distillation.modal_vllm::parity --checkpoint <run/step-N.pt>

`curriculum` recipes sample a pinned teacher on one H100 and verify every answer
(`curriculum.py`); `parity` exports a distillation checkpoint to Llama format and compares
vLLM with Transformers on one L4. Each launch makes one call with no retries. The launcher
records a deadline 24 hours after launch and a call refuses to start unless its whole
function timeout fits before it. Outputs go to the `lm-exploration-distillation` volume.
"""

import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
CURRICULUM_TIMEOUT = 2 * 60 * 60
PARITY_TIMEOUT = 15 * 60
PARITY_PROMPTS = [
    "def add(a, b):\n    return",
    "class Stack:\n    def __init__(self):",
    "for i in range(10):\n    print(",
    "# café\nx = ",
    "function sum(a, b) {\n  return",
]
INSTRUCTION = (
    'Return only a JSON object with string keys "code" and "explanation". '
    "Code must define exactly one function named solve, with no imports, decorators, type "
    "annotations, helper functions or top-level execution. "
    "Use basic Python loops, arithmetic, indexing and builtins only. Available builtins: abs, "
    "all, any, bool, dict, enumerate, int, len, list, max, min, range, reversed, set, sorted, "
    "sum, tuple, zip. "
    "Allowed methods: append, add, get, count, index, reverse, sort, copy. Do not mutate inputs. "
    "Explain the mathematical meaning and why the algorithm works in 2-4 sentences, including "
    "a small worked example and an edge case. "
    "Keep code and explanation concise. No markdown fences around JSON."
)

app = modal.App("lm-distillation-vllm", include_source=False)
volume = modal.Volume.from_name("lm-exploration-distillation")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "tokenizers==0.22.2")
    .env({"PYTHONPATH": "/project", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    .add_local_dir(
        "distillation", "/project/distillation", ignore=["artifacts", "__pycache__"]
    )
    .add_local_dir("training/100m", "/project/model_100m")
    .add_local_dir("training/10m", "/project/model_10m")
    .add_local_dir(
        "training/data", "/project/training_data", ignore=["eu", "__pycache__"]
    )
    .add_local_dir(
        "training/tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1"
    )
)


def require_time(deadline, timeout):
    if time.time() + timeout > deadline:
        raise TimeoutError(
            "The call's timeout no longer fits before the launch deadline"
        )


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=CURRICULUM_TIMEOUT,
    retries=0,
    volumes={"/distill": volume},
    max_containers=1,
)
def curriculum(recipe, deadline):
    import hashlib
    import os
    import shutil
    from collections import Counter

    os.environ["HF_HOME"] = "/distill/hf-cache"
    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distillation.curriculum import task, training_text, verify
    from distillation.recipe import validate_recipe

    spec = validate_recipe(recipe)
    require_time(deadline, CURRICULUM_TIMEOUT)
    teacher, revision = spec["teacher"]["repo"], spec["teacher"]["revision"]
    sampling, count = spec["sampling"], spec["count"]
    started = time.monotonic()
    output = Path("/distill") / spec["run_id"]
    output.mkdir(exist_ok=False)
    shutil.copytree(
        "/project/distillation",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__", "artifacts"),
    )
    settings = {
        "teacher": teacher,
        "revision": revision,
        "requested_examples": count,
        **sampling,
        "thinking": False,
        "student_tokenizer_sha256": TOKENIZER_SHA256,
    }
    (output / "request.json").write_text(json.dumps(settings, indent=2))
    tasks = [task(i) for i in range(count)]
    if len({t["id"] for t in tasks}) != count:
        raise ValueError("Task specifications repeat; expand curriculum before scaling")
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    volume.commit()
    tokenizer = AutoTokenizer.from_pretrained(teacher, revision=revision)
    frozen = load_frozen()
    llm = LLM(
        model=teacher,
        revision=revision,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        seed=sampling["seed"],
        enable_prefix_caching=True,
    )
    params = SamplingParams(**sampling)
    counters = Counter()
    hashes = set()
    generation_started = time.monotonic()
    progress = dict(counters)
    for offset in range(0, count, 32):
        batch = tasks[offset : offset + 32]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": INSTRUCTION},
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
        scope="bounded template pilot; not a large corpus or proof of scaling",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=PARITY_TIMEOUT,
    retries=0,
    max_containers=1,
    volumes={"/distill": volume},
)
def qualify(checkpoint, deadline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from distillation.export import export_checkpoint
    from distillation.recipe import checkpoint as checked

    checked(checkpoint)
    require_time(deadline, PARITY_TIMEOUT)
    export = Path("/distill") / (checkpoint.removesuffix(".pt") + "-hf")
    report = export_checkpoint("/distill/" + checkpoint, export)
    tokenizer = AutoTokenizer.from_pretrained(export)
    prefix = tokenizer.bos_token_id
    if prefix is None:
        prefix = tokenizer.eos_token_id
    token_ids = [
        [prefix] + tokenizer.encode(p, add_special_tokens=False) for p in PARITY_PROMPTS
    ]
    reference = AutoModelForCausalLM.from_pretrained(
        export, torch_dtype=torch.float32
    ).eval()
    expected = []
    with torch.no_grad():
        for ids in token_ids:
            logits = reference(torch.tensor([ids])).logits[0, -1].float()
            token = logits.argmax().item()
            expected.append((token, logits.log_softmax(-1)[token].item()))
    del reference
    llm = LLM(
        model=str(export),
        dtype="float32",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids],
        SamplingParams(temperature=0, max_tokens=1, logprobs=1),
    )
    errors = []
    for result, (token, logprob) in zip(outputs, expected, strict=True):
        actual = result.outputs[0]
        if actual.token_ids != [token]:
            raise ValueError("vLLM greedy token mismatch")
        error = abs(actual.logprobs[0][token].logprob - logprob)
        if error > 1e-3:
            raise ValueError(f"vLLM log probability mismatch: {error}")
        errors.append(error)
    report.update(
        vllm_version="0.11.0",
        prompts=len(PARITY_PROMPTS),
        max_logprob_error=max(errors),
        export=str(export),
        dtype="float32",
    )
    (export / "vllm-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))


@app.local_entrypoint()
def main(recipe: str, run_id: str = "", output: str = ""):
    from distillation.recipe import load_recipe

    spec = load_recipe(recipe, run_id)
    if spec["kind"] != "curriculum":
        raise ValueError(f"{spec['kind']} recipes run with distillation.modal_app")
    result = curriculum.remote(spec, time.time() + LAUNCH_LIMIT_SECONDS)
    write(
        output or f"distillation/artifacts/{spec['run_id']}.json",
        {"recipe": spec, "result": result},
    )


@app.local_entrypoint()
def parity(checkpoint: str, output: str = ""):
    from distillation.recipe import checkpoint as checked

    checked(checkpoint)
    result = qualify.remote(checkpoint, time.time() + LAUNCH_LIMIT_SECONDS)
    name = checkpoint.removesuffix(".pt").replace("/", "-")
    write(output or f"distillation/artifacts/{name}-vllm-parity.json", result)
