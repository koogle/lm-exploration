"""Standard benchmarks (lm-evaluation-harness) for a hybrid export and reference models.

Run from training/:
    modal run --detach -m model_hybrid.modal_benchmarks [--exports tokens-A,tokens-B] [--models M,...]

One H100 call per model, in parallel: the hybrid export through `benchmarks.py`
and each reference model through the harness's Hugging Face backend, all with the
same tasks, shot counts and greedy decoding. Each call has `retries=0`, a 6-hour
timeout and refuses to start 24 hours after launch. Code tasks execute the
generated programs inside the call's container. Each call writes its results
(harness JSON without per-sample logs) to
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
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/runs": runs_volume},
)
def evaluate(model, revision, deadline, folder, limit=0):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import lm_eval
    import torch

    torch._dynamo.config.cache_size_limit = 64
    if model.startswith("hybrid:"):
        from model_hybrid import benchmarks

        harness_model = benchmarks.load(Path("/runs") / RUN / "export" / model.split(":", 1)[1])
    else:
        from lm_eval.models.huggingface import HFLM

        harness_model = HFLM(pretrained=model, revision=revision, dtype="bfloat16", batch_size=8)
    out = {"model": model, "revision": revision, "limit": limit or None, "suites": []}
    for tasks, shots in SUITES:
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
        )
        out["suites"].append(
            {
                "tasks": tasks,
                "num_fewshot": shots,
                "seconds": round(time.time() - started, 1),
                "summary": _summary(results),
                "results": results["results"],
                "configs": results["configs"],
                "versions": results["versions"],
                "n-samples": results["n-samples"],
            }
        )
        print(json.dumps({"model": model, "tasks": tasks, **out["suites"][-1]["summary"]}))
    # Plain JSON (the harness's configs hold its own objects), also on the volume so a
    # detached launch keeps every finished model.
    text = json.dumps(out, indent=2, default=str) + "\n"
    path = Path("/runs/benchmarks", folder, f"{model.replace('/', '--')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    runs_volume.commit()
    return json.loads(text)


@app.function(image=image, cpu=0.25, memory=512, timeout=86400, retries=0)
def enforce_deadline(call_ids, deadline):
    """Independent supervisor: cancel every call of this launch still running at its deadline."""
    while (remaining := deadline - time.time()) > 0:
        time.sleep(min(remaining, 300))
    for call_id in call_ids:
        modal.FunctionCall.from_id(call_id).cancel()
    return {"cancelled": call_ids}


@app.local_entrypoint()
def main(exports: str = "tokens-039998980096", limit: int = 0, models: str = ""):
    """Spawn one evaluation per model; results land on the runs volume as each finishes.

    `--exports` (comma-separated) names hybrid exports; `--models` (comma-separated,
    "hybrid" for the exports) limits the launch to those. Calls are spawned, not
    mapped, so they keep running after this client exits.
    """
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    chosen = [(f"hybrid:{e}", None) for e in exports.split(",")] + list(REFERENCES.items())
    if models:
        wanted = set(models.split(","))
        chosen = [(m, r) for m, r in chosen if m in wanted or (r is None and "hybrid" in wanted)]
    run_folder = f"{int(time.time())}{'-limit' if limit else ''}"
    calls = [evaluate.spawn(m, r, deadline, run_folder, limit) for m, r in chosen]
    supervisor = enforce_deadline.spawn([c.object_id for c in calls], deadline)
    print(
        json.dumps(
            {
                "results": f"lm-exploration-hybrid-runs/benchmarks/{run_folder}/",
                "calls": {m: c.object_id for (m, _), c in zip(chosen, calls, strict=True)},
                "supervisor": supervisor.object_id,
                "deadline_unix": deadline,
            },
            indent=1,
        )
    )
