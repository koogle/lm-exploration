"""Benchmark training checkpoints from the run volume on one Modal H100.

Run from training/:
    modal run -m evaluation.modal_evaluate --name NAME --snapshot ROOT/VERSION/snapshots/RUN \
        --checkpoints RUN-sNN/step-N.pt[,RUN-sMM/step-M.pt ...] [--config benchmarks.json] \
        [--shots 5] [--tasks gsm8k,svamp] [--output result.json]

`--config` names the benchmark datasets (pinned HF commits) and settings; checkpoints load
with their runs' own model code (`trainer/checkpoints.py`). One function call, a 3-hour hard timeout
and no retries; the call refuses new work 24 hours after launch, including time spent
queued. Results go to `/runs/evaluation/NAME.json` after each checkpoint and to `--output`
(default `artifacts/evaluation/NAME.json`).
"""

import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
TIMEOUT_SECONDS = 3 * 60 * 60
DEFAULT_CONFIG = "analysis/evaluation/configs/benchmarks.json"
# Benchmarks that show worked exemplars before each problem.
MATH = ("gsm8k", "svamp")
TASKS = (
    "heldout",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "winogrande",
    "lambada",
    *MATH,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "pyarrow==26.0.0",
        "math-verify==0.8.0",
        "pydantic>=2.11,<3",
        "tokenizers==0.22.2",
        "huggingface-hub==1.33.0",
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("analysis/evaluation", "/project/evaluation", ignore=["__pycache__"])
    .add_local_file("trainer/rollout.py", "/project/trainer/rollout.py")
    .add_local_file("trainer/__init__.py", "/project/trainer/__init__.py")
    .add_local_file("trainer/checkpoints.py", "/project/trainer/checkpoints.py")
)
app = modal.App("checkpoint-evaluation")
run_volume = modal.Volume.from_name("lm-exploration-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing further evaluation")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config["datasets"]) != set(TASKS) - {"heldout"}:
        raise ValueError(f"datasets must cover {TASKS[1:]}")
    for task in MATH:
        if "exemplars" not in config["datasets"][task]:
            raise ValueError(f"{task} needs an exemplars file")
    return config


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume, "/data": data_volume},
)
def evaluate(name, snapshot, checkpoints, config, deadline, shots, selected):
    import torch
    from training_data.hub_files import pinned_rows
    from training_data.subword import digest

    from evaluation import heldout, scoring
    from evaluation.benchmarks import BENCHMARKS
    from evaluation.checkpoints import scorer
    from trainer.checkpoints import load

    check_deadline(deadline)
    output = Path("/runs/evaluation") / f"{name}.json"
    result = {
        "name": name,
        "snapshot": snapshot,
        "config": config,
        "settings": {"shots": shots, "tasks": selected},
        "source_sha256": {
            str(path.relative_to("/project")): digest(path)
            for path in sorted(Path("/project/evaluation").rglob("*.py"))
        },
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "models": [],
    }
    data = {}
    for task in selected:
        if task != "heldout":
            spec = config["datasets"][task]
            exemplars = spec.get("exemplars")
            data[task] = (
                pinned_rows(spec["repo"], spec["revision"], spec["file"]),
                pinned_rows(spec["repo"], spec["revision"], exemplars) if exemplars else [],
            )
    windows = None
    for checkpoint in checkpoints:
        check_deadline(deadline)
        started = time.time()
        loaded = load(Path("/runs") / checkpoint, device="cuda")
        model = scorer(loaded)
        info = {"checkpoint": checkpoint, **loaded.info}
        print(json.dumps(info), flush=True)
        scores, answers = {}, {}
        if "heldout" in selected:
            if windows is None:
                windows, result["snapshot_weights"] = heldout.windows(
                    Path("/data") / snapshot,
                    model,
                    loaded.tokenizer,
                    config["heldout_windows_per_source"],
                )
            scores["heldout"] = {s: scoring.bits_per_byte(model, w) for s, w in windows.items()}
        for task, (rows, exemplars) in data.items():
            check_deadline(deadline)
            settings = config["datasets"][task]
            benchmark = BENCHMARKS[task]
            scores[task], generated = benchmark.score(model, rows, exemplars, settings, shots)
            if generated is not None:
                answers[task] = generated
        result["models"].append(
            {**info, "scores": scores, "answers": answers, "seconds": time.time() - started}
        )
        print(json.dumps({"checkpoint": checkpoint, "scores": scores}), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        run_volume.commit()
        del loaded, model
        torch.cuda.empty_cache()
    return result


@app.local_entrypoint()
def main(
    name: str,
    snapshot: str,
    checkpoints: str,
    config: str = DEFAULT_CONFIG,
    output: str = "",
    shots: int = 5,
    tasks: str = "",
):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    if not 0 <= shots <= 8:
        raise ValueError("shots must be from 0 to 8")
    selected = tasks.split(",") if tasks else list(TASKS)
    if set(selected) - set(TASKS):
        raise ValueError(f"tasks must be among {TASKS}")
    benchmarks = load_config(config)
    result = evaluate.remote(
        name, snapshot, checkpoints.split(","), benchmarks, deadline, shots, selected
    )
    path = Path(output or f"artifacts/evaluation/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))
