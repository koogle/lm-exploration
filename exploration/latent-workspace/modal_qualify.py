"""One-time frozen-checkpoint longer-rollout and sealed-test qualification."""

import json
from pathlib import Path

import modal

from latent_workspace.budget import reserve

app = modal.App("latent-workspace-qualification")
cache = modal.Volume.from_name("latent-workspace-qwen-cache")
runs = modal.Volume.from_name("latent-workspace-qwen-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.12")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .add_local_file("qualification_worker.py", "/project/qualification_worker.py", copy=True)
    .run_commands("cd /project && uv sync --frozen --no-install-project")
    .env({"PATH": "/project/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .add_local_python_source("latent_workspace")
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=3600,
    retries=0,
    single_use_containers=True,
    volumes={"/runs": runs, "/cache": cache},
)
def qualification(protocol, run_id):
    import sys

    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from qualification_worker import qualify

    runs.reload()
    try:
        return qualify(protocol, f"/runs/{run_id}", runs.commit)
    finally:
        runs.commit()
        cache.commit()


@app.local_entrypoint()
def main(protocol: str, run_id: str):
    cfg = json.loads(Path(protocol).read_text())
    reserve("artifacts/budget.json", run_id, "L40S", timeout_seconds=3600)
    call = qualification.spawn(cfg, run_id)
    Path("artifacts/qualification-handle.json").write_text(
        json.dumps({"run_id": run_id, "call_id": call.object_id}, indent=2) + "\n"
    )
    print(call.object_id, flush=True)
