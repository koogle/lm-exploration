"""Standard benchmarks (lm-evaluation-harness) for a hybrid export and reference models.

Run from training/:
    modal run -m model_hybrid.modal_benchmarks [--export tokens-039998980096] [--limit N]

One H100 call per model, in parallel: the hybrid export through `benchmarks.py`
and each reference model through the harness's Hugging Face backend, all with the
same tasks, shot counts and greedy decoding. Each call has `retries=0`, a 3-hour
timeout and refuses to start 24 hours after launch. Code tasks execute the
generated programs inside the call's container. Results (harness JSON without
per-sample logs) are written to the ignored artifacts/results/.
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
        "lm-eval==0.4.9.1",
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "datasets==3.6.0",
        "evaluate==0.4.5",
    ),
    {"HF_ALLOW_CODE_EVAL": "1", "HF_DATASETS_TRUST_REMOTE_CODE": "1"},
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
    timeout=3 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/runs": runs_volume},
)
def evaluate(model, revision, deadline, limit=0):
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

        harness_model = HFLM(pretrained=model, revision=revision, dtype="bfloat16", batch_size=32)
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
    return out


@app.local_entrypoint()
def main(export: str = "tokens-039998980096", limit: int = 0, references: bool = True):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    models = [(f"hybrid:{export}", None)]
    if references:
        models += list(REFERENCES.items())
    stamp = int(time.time())
    folder = Path("artifacts/results/benchmarks")
    folder.mkdir(parents=True, exist_ok=True)
    for result in evaluate.starmap([(m, r, deadline, limit) for m, r in models]):
        name = result["model"].replace("hybrid:", f"{RUN}-").replace("/", "--")
        path = folder / f"{name}-{stamp}{'-limit' if limit else ''}.json"
        path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(path)
