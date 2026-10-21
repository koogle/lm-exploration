"""A single bounded H100 allocation; no autoscaling, retries or student training."""

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import modal

from core import atomic_json, reserve_budget, validate_config

HERE = Path(__file__).parent if modal.is_local() else Path("/experiment")
CONFIG = json.loads((HERE / "config.json").read_text())
app = modal.App("lm-exploration-teacher-rollouts")
image = (
    modal.Image.from_registry(CONFIG["image"])
    .entrypoint([])
    .pip_install_from_requirements(str(HERE / "requirements.lock"))
    .add_local_dir(
        HERE,
        "/experiment",
        ignore=["artifacts", ".budget", "__pycache__", ".venv", ".ruff_cache"],
    )
)
volume = modal.Volume.from_name(
    "lm-exploration-teacher-rollouts", create_if_missing=True
)


@app.function(
    image=image,
    gpu="H100",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def worker(run_id, attempt, manifest):
    c = manifest["config"]
    validate_config(c)
    if not c["budget_authorized"] or not c["authorization_reference"]:
        raise ValueError("budget not authorized")
    for name in (run_id, attempt):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("invalid ID")
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"input-{attempt}.json"
    if path.exists():
        raise ValueError("attempt already used")
    atomic_json(path, manifest)
    volume.commit()
    started = time.monotonic()
    status = "failed"
    try:
        # Parent timeout includes startup, model loading, generation and verification.
        proc = subprocess.Popen(
            ["python", "/experiment/runner.py", str(root), attempt, str(path)],
            cwd="/experiment",
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=c["runtime_seconds"] - 45)
        except subprocess.TimeoutExpired:
            status = "timeout"
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        if returncode:
            raise RuntimeError(f"runner failed with exit code {returncode}")
        status = "completed"
    finally:
        atomic_json(
            root / f"worker-{attempt}.json",
            {"elapsed_seconds": time.monotonic() - started, "status": status},
        )
        volume.commit()
    return json.loads((root / "attempts" / attempt / "report.json").read_text())


@app.local_entrypoint()
def main(run_id: str, attempt: str, config: str = "config.json", launch: bool = False):
    c = json.loads(Path(config).read_text())
    validate_config(c)
    for key in ("image", "runtime", "model", "revision", "dtype"):
        if c[key] != CONFIG[key]:
            raise ValueError(f"{key} differs from pinned build config")
    for name in (run_id, attempt):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("invalid ID")
    prompts = json.loads((HERE / "prompts.json").read_text())
    manifest = {
        "config": c,
        "prompts": prompts,
        "tokenizer_revision": c["revision"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.glob("*.py"))
            + [HERE / "prompts.json", HERE / "requirements.lock"]
        },
    }
    if not launch:
        print(json.dumps({"mode": "dry_run", "manifest": manifest}, indent=2))
        return
    reserve_budget(c, HERE / ".budget", attempt)
    result = worker.with_options(timeout=c["runtime_seconds"]).remote(
        run_id, attempt, manifest
    )
    out = HERE / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"{attempt}-report.json", result)
    print(json.dumps(result, indent=2))
