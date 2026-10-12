"""Bounded resumable long-response preparation; shares the authoritative ledger."""

import json
import re
from pathlib import Path

import modal

from latent_workspace.budget import reserve

app = modal.App("latent-workspace-qwen-long-data")
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


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache, "/runs": runs},
)
def prepare(config, run_id, manifest_parent, seed_parent, resume_parent):
    import sys

    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.dflash import DraftConfig
    from latent_workspace.dflash_large_data import prepare_chunk, prepare_manifest

    try:
        if not manifest_parent:
            return prepare_manifest(
                DraftConfig(**config),
                f"/runs/{run_id}",
                f"/runs/{seed_parent}",
                on_checkpoint=runs.commit,
            )
        return prepare_chunk(
            DraftConfig(**config),
            f"/runs/{run_id}",
            f"/runs/{manifest_parent}",
            f"/runs/{seed_parent}",
            f"/runs/{resume_parent}" if resume_parent else "",
            on_checkpoint=runs.commit,
        )
    finally:
        runs.commit()
        cache.commit()


@app.local_entrypoint()
def main(
    config: str,
    run_id: str,
    manifest_parent: str = "",
    seed_parent: str = "dflash-data-54k-01",
    resume_parent: str = "",
):
    from latent_workspace.dflash import DraftConfig

    cfg = DraftConfig(**json.loads(Path(config).read_text())).validate()
    for name in (run_id, seed_parent, manifest_parent, resume_parent):
        if name and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name):
            raise ValueError("invalid run ID")
    if cfg.max_seconds > 3300 or (resume_parent and not manifest_parent):
        raise ValueError("invalid bounded preparation")
    amount = reserve("artifacts/budget.json", run_id, "L40S", timeout_seconds=3600)
    print(f"Reserved ${amount:.2f} for bounded preparation.")
    report = prepare.remote(cfg.to_dict(), run_id, manifest_parent, seed_parent, resume_parent)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
