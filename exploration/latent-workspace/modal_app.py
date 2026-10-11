"""Bounded single-GPU jobs; no deployed service or automatic retries."""

import json
import re
from pathlib import Path

import modal

from latent_workspace.budget import CAP_USD, TIMEOUT_SECONDS, reserve
from latent_workspace.config import Config

app = modal.App("latent-workspace-qwen")
cache = modal.Volume.from_name("latent-workspace-qwen-cache", create_if_missing=True)
runs = modal.Volume.from_name("latent-workspace-qwen-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.12")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .run_commands("cd /project && uv sync --frozen --no-install-project")
    .env({"PATH": "/project/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .add_local_python_source("latent_workspace")
)


def execute(config, run_id, resume_from):
    import sys

    # Modal's Python worker is external to the uv venv; use locked dependencies.
    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.train import run

    cfg = Config(**config).validate()
    output = f"/runs/{resume_from or run_id}"
    try:
        if cfg.study_layer_controls:
            from latent_workspace.small_controls import run_controls

            report = run_controls(cfg, output, on_checkpoint=runs.commit)
        elif cfg.study_eval_only:
            from latent_workspace.posthoc import evaluate_study

            report = evaluate_study(cfg, output)
        elif cfg.study_sizes:
            from latent_workspace.study import run_study

            report = run_study(cfg, output, on_checkpoint=runs.commit)
        else:
            report = run(cfg, output, resume=bool(resume_from), on_checkpoint=runs.commit)
        return {"run_id": run_id, "output": output, "report": report}
    finally:
        runs.commit()
        cache.commit()


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache, "/runs": runs},
)
def l40s(config: dict, run_id: str, resume_from: str = ""):
    return execute(config, run_id, resume_from)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=131072,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache, "/runs": runs},
)
def h100(config: dict, run_id: str, resume_from: str = ""):
    return execute(config, run_id, resume_from)


@app.local_entrypoint()
def main(
    config: str = "configs/smoke.json",
    run_id: str = "smoke-l40s",
    gpu: str = "L40S",
    resume_from: str = "",
):
    cfg = Config.load(config)
    if cfg.max_seconds > TIMEOUT_SECONDS - 300:
        raise ValueError("max_seconds must leave 300s for checkpoint/evaluation cleanup")
    for identifier in (run_id, resume_from):
        if identifier and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identifier):
            raise ValueError("invalid run identifier")
    if gpu not in ("L40S", "H100"):
        raise ValueError("choose L40S or H100")
    amount = reserve("artifacts/budget.json", run_id, gpu)
    print(f"Reserved ${amount:.2f} against the ${CAP_USD:g} local allocation cap.")
    function = l40s if gpu == "L40S" else h100
    result = function.remote(cfg.to_dict(), run_id, resume_from)
    destination = Path("results") / f"{run_id}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {destination}; checkpoints in Modal volume {result['output']}")
