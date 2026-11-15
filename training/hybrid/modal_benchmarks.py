"""Standard benchmarks (lm-evaluation-harness) for a hybrid export and reference models.

Run from training/:
    modal run --detach -m model_hybrid.modal_benchmarks [--exports tokens-A,tokens-B] [--models M,...]
        [--suite standard|code]

`--suite code` runs HumanEval (as shipped and with stripped prompts, `fim_tasks.py`), MBPP and HumanEval-Infilling (single- and multi-line) in
each model's own fill-in-the-middle format (`fim_tasks.py`) for the hybrid exports
(GLM code tokens) and the code reference models.

One H100 call per model, in parallel: the hybrid export through `benchmarks.py`
and each reference model through the harness's Hugging Face backend, all with the
same tasks, shot counts and greedy decoding. Each call has `retries=0`, a 12-hour
timeout and refuses to start 24 hours after launch. Code tasks execute the
generated programs inside the call's container. Each call writes its results
(harness JSON) and every prompt, response and per-sample score (`.samples.jsonl`) to
`lm-exploration-hybrid-runs/benchmarks/<launch>/`; an independent supervisor cancels
any call still running 24 hours after launch.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid.image import LAUNCH_LIMIT_SECONDS, build

app = modal.App("lm-exploration-hybrid-benchmarks")
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs")
secret = modal.Secret.from_name("simple-llm-hf")
image = build(
    (
        "lm-eval==0.4.13",
        "transformers==5.19.0",
        "accelerate==1.15.0",
        "datasets==4.8.5",
        "evaluate==0.4.6",
    ),
    {"HF_ALLOW_CODE_EVAL": "1"},
)
RUN = "hybrid-720m-40b-01"
# (tasks, few-shot examples); None keeps the task's own default.
SUITES = (
    (
        [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
            "winogrande",
            "openbookqa",
            "sciq",
            "lambada_openai",
        ],
        0,
    ),
    (["mmlu"], 5),
    (["gsm8k"], 5),
    (["humaneval"], 0),
    (["mbpp"], 3),
)
# Code models for `--suite code`: revision and fill-in-the-middle format.
CODE_REFERENCES = {
    "bigcode/gpt_bigcode-santacoder": ("291931872cae83498cf984b16319f47f5e9e7a07", "santacoder"),
    "deepseek-ai/deepseek-coder-1.3b-base": (
        "c919139c3a9b4070729c8b2cca4847ab29ca8d94",
        "deepseek",
    ),
    "Qwen/Qwen2.5-Coder-0.5B": ("8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301", "qwen"),
    "Qwen/Qwen2.5-Coder-1.5B": ("df3ce67c0e24480f20468b6ef2894622d69eb73b", "qwen"),
}
# Reference models trained with a beginning-of-sequence token that their tokenizer, as
# transformers loads it, does not add. Without it DeepSeek-Coder infills the wrong line.
BOS_MODELS = {"deepseek-ai/deepseek-coder-1.3b-base"}
REFERENCES = {
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "LiquidAI/LFM2.5-1.2B-Base": "7453bca97ca1e67754c4035a4b4c584e1c9dd725",
    "allenai/OLMo-2-0425-1B": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    "google/gemma-3-270m": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "google/gemma-3-1b-pt": "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
    "google/gemma-4-E2B": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
}


def _summary(results):
    """Task -> "metric,filter" -> value, without the harness's stderr and alias entries."""
    return {
        task: {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and "_stderr" not in key
        }
        for task, metrics in results["results"].items()
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/runs": runs_volume},
)
def evaluate(model, revision, deadline, folder, limit=0, fim_format=None, part=""):
    """Every suite for one model; with `fim_format`, the code suite in that format, or
    only its `part` (one of `CODE_PARTS`)."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import lm_eval
    import torch

    torch._dynamo.config.cache_size_limit = 64
    if model.startswith("hybrid:"):
        from model_hybrid import benchmarks

        export = model.split(":", 1)[1]
        run, export = export.split("/") if "/" in export else (RUN, export)
        harness_model = benchmarks.load(Path("/runs") / run / "export" / export)
    else:
        from lm_eval.models.huggingface import HFLM

        harness_model = HFLM(
            pretrained=model,
            revision=revision,
            dtype="bfloat16",
            batch_size=8,
            add_bos_token=True if model in BOS_MODELS else None,
        )
    out = {"model": model, "revision": revision, "limit": limit or None, "suites": []}
    suites = SUITES
    if fim_format:
        from model_hybrid.fim_tasks import humaneval_config, task_configs

        single, multi = task_configs(fim_format)
        parts = {
            "left_to_right": ((["humaneval", humaneval_config()], 0), (["mbpp"], 3)),
            "single_line": (([single], 0),),
            "multi_line": (([multi], 0),),
        }
        suites = parts[part] if part else (*parts["left_to_right"], ([single, multi], 0))
        out["part"] = part or None
        out["fim_format"] = fim_format
    samples = []
    for tasks, shots in suites:
        started = time.time()
        results = lm_eval.simple_evaluate(
            model=harness_model,
            tasks=tasks,
            num_fewshot=shots,
            limit=limit or None,
            confirm_run_unsafe_code=True,
            random_seed=1234,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
            log_samples=True,
        )
        for task, rows in results.pop("samples").items():
            for row in rows:
                samples.append(
                    {
                        "task": task,
                        "doc_id": row["doc_id"],
                        "prompt": row["arguments"][0][0],
                        "response": row["resps"][0][0],
                        "filtered": row["filtered_resps"][0],
                        **{m: row[m] for m in row["metrics"]},
                    }
                )
        out["suites"].append(
            {
                "tasks": [t["task"] if isinstance(t, dict) else t for t in tasks],
                "num_fewshot": shots,
                "seconds": round(time.time() - started, 1),
                "summary": _summary(results),
                "results": results["results"],
                "configs": results["configs"],
                "versions": results["versions"],
                "n-samples": results["n-samples"],
            }
        )
        print(json.dumps({"model": model, **out["suites"][-1]["summary"]}))
    # Plain JSON (the harness's configs hold its own objects), also on the volume so a
    # detached launch keeps every finished model.
    text = json.dumps(out, indent=2, default=str) + "\n"
    name = model.replace("/", "--") + (f".{part}" if part else "")
    path = Path("/runs/benchmarks", folder, f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    with path.with_suffix(".samples.jsonl").open("w") as handle:
        for row in samples:
            handle.write(json.dumps(row, default=str) + "\n")
    runs_volume.commit()
    return json.loads(text)


@app.function(image=image, cpu=0.25, memory=512, timeout=86400, retries=0)
def enforce_deadline(call_ids, deadline):
    """Independent supervisor: cancel every call of this launch still running at its deadline."""
    from workflows.deadline import calls_finished, cancel_at_deadline

    cancelled = cancel_at_deadline(deadline, lambda: call_ids, lambda: calls_finished(call_ids))
    return {"stage": "completed"} if cancelled is None else {"cancelled": cancelled}


@app.local_entrypoint()
def main(
    exports: str = "tokens-039998980096",
    limit: int = 0,
    models: str = "",
    suite: str = "standard",
    split: str = "",
):
    """Spawn one evaluation per model; results land on the runs volume as each finishes.

    `--exports` (comma-separated) names hybrid exports of the 40B run, or of another run
    as `<run id>/<export>`; `--models` (comma-separated,
    "hybrid" for the exports) limits the launch to those. Calls are spawned, not
    mapped, so they keep running after this client exits. `--split all` runs the code
    suite as one call per part (left-to-right, single-line, multi-line infilling), each
    writing `<model>.<part>.json`, for a model that needs more than one call's time;
    `--split multi_line` (comma-separated part names) runs only those parts.
    """
    if suite not in ("standard", "code"):
        raise ValueError("--suite is standard or code")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    hybrid = "glm" if suite == "code" else None
    chosen = [(f"hybrid:{e}", None, hybrid) for e in exports.split(",")]
    if suite == "code":
        chosen += [(m, r, f) for m, (r, f) in CODE_REFERENCES.items()]
    else:
        chosen += [(m, r, None) for m, r in REFERENCES.items()]
    if models:
        wanted = set(models.split(","))
        chosen = [c for c in chosen if c[0] in wanted or (c[1] is None and "hybrid" in wanted)]
    run_folder = f"{int(time.time())}-{suite}{'-limit' if limit else ''}"
    code_parts = ("left_to_right", "single_line", "multi_line")
    parts = ("",)
    if split and suite == "code":
        parts = code_parts if split == "all" else tuple(split.split(","))
        if not set(parts) <= set(code_parts):
            raise ValueError(f"--split is all or a comma-separated subset of {code_parts}")
    chosen = [(m, r, f, part) for m, r, f in chosen for part in parts]
    calls = [evaluate.spawn(m, r, deadline, run_folder, limit, f, p) for m, r, f, p in chosen]
    supervisor = enforce_deadline.spawn([c.object_id for c in calls], deadline)
    print(
        json.dumps(
            {
                "results": f"lm-exploration-hybrid-runs/benchmarks/{run_folder}/",
                "calls": {
                    f"{m}{'.' + p if p else ''}": c.object_id
                    for (m, _, _, p), c in zip(chosen, calls, strict=True)
                },
                "supervisor": supervisor.object_id,
                "deadline_unix": deadline,
            },
            indent=1,
        )
    )
