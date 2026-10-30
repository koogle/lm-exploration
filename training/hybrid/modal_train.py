"""Hybrid pretraining, benchmarks and pilots on Modal H100s.

Run from training/:
    modal run -m model_hybrid.modal_train::bench --gpus 1 [--microbatch 4]
    modal run --detach -m model_hybrid.modal_train::launch --run-id <id> \\
        [--config hybrid/configs/hybrid-720m-40b.json] [--max-tokens N]
    modal run --detach -m model_hybrid.modal_train::launch_rl --run-id <id> --config <rl config>

`launch` fixes one deadline 24 hours ahead (less a supervisor margin), starts an
independent supervisor that cancels every recorded call at that deadline, then
starts one 8xH100 training segment (`retries=0`, timeout bounded by the deadline).
The segment stops itself SHUTDOWN_SECONDS before its timeout and saves a
checkpoint; launching again with the same run ID resumes from the latest
checkpoint in `lm-exploration-hybrid-runs/<run id>/`. Training reads prepared
shards from `lm-exploration-hybrid-data` and waits for shards still being
prepared. `launch_rl` runs model_hybrid.rl the same way (one supervised 8xH100
segment, resumable, and extendable by relaunching with a larger `steps`), in an
image that also grades completions. Each benchmark is
one call with a 30-minute timeout and no retries.
"""

import io
import json
import subprocess
import threading
import time
from pathlib import Path

import modal
from model_hybrid.image import build, image

app = modal.App("lm-exploration-hybrid-train")
data_volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs", create_if_missing=True)
secret = modal.Secret.from_name("simple-llm-hf")
# run id -> "stop" asks a running segment to checkpoint and exit at the next step.
control = modal.Dict.from_name("lm-exploration-hybrid-control", create_if_missing=True)
STOP_FILE = "/tmp/hybrid-stop"
DEFAULT_CONFIG = "hybrid/configs/hybrid-720m-40b.json"
SHUTDOWN_SECONDS = 900  # Final checkpoint (~7 GB) and exit.
VOLUMES = {"/data": data_volume, "/runs": runs_volume}
# RL also grades completions: math_verify, and pytest for generated code.
rl_image = build(("math-verify==0.9.0", "pytest==8.4.2"))


def run_torchrun(module, args, gpus, timeout, log_path, run_id=""):
    """Run torchrun, tee its output to log_path, and keep both volumes in sync meanwhile.

    With a run id, a "stop" entry in the control Dict creates the trainer's stop file.
    """
    command = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", module, *args]
    stop = threading.Event()

    def sync():
        while not stop.wait(60):
            if run_id and control.get(run_id) == "stop":
                Path(STOP_FILE).touch()
            for action in (data_volume.reload, runs_volume.commit):
                try:
                    action()
                except Exception as error:  # noqa: BLE001 - retried on the next pass.
                    print(f"volume sync: {error}", flush=True)

    worker = threading.Thread(target=sync, daemon=True)
    worker.start()
    lines = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        process = subprocess.Popen(
            command, cwd="/project", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        start = time.monotonic()
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
            if time.monotonic() - start > timeout:
                process.terminate()
        code = process.wait()
    stop.set()
    runs_volume.commit()
    return code, lines


def run_segment(
    run_id, config, timeout, deadline, gpus, provenance, module, extra_args=(), extendable=()
):
    """One torchrun segment of `module` for a run folder, bounded by the launch deadline.

    A run resumes only with its original config, except for `extendable` keys (such
    as an RL run's step count, which does not change its schedule).
    """
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    started = time.time()
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "run-config.json"
    if config_path.exists():
        original = json.loads(config_path.read_text())
        fixed = {k: v for k, v in original.items() if k not in extendable}
        if fixed != {k: v for k, v in config.items() if k not in extendable}:
            raise ValueError("A run resumes only with its original config")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    launch = {
        **(provenance or {}),
        "started_unix": started,
        "gpus": gpus,
        "deadline_unix": deadline,
    }
    (folder / "launches").mkdir(exist_ok=True)
    (folder / "launches" / f"{int(started)}.json").write_text(json.dumps(launch, indent=2) + "\n")
    runs_volume.commit()
    stop_unix = min(deadline, started + timeout) - SHUTDOWN_SECONDS
    args = [
        "--config",
        str(config_path),
        "--run-dir",
        str(folder),
        "--data-root",
        "/data",
        "--stop-unix",
        str(stop_unix),
        *extra_args,
    ]
    code, _ = run_torchrun(
        module, args, gpus, timeout - 300, folder / "logs" / f"segment-{int(started)}.log", run_id
    )
    report = folder / ("summary.json" if module == "model_hybrid.rl" else "report.json")
    result = json.loads(report.read_text()) if report.exists() else {}
    if code:
        raise RuntimeError(f"Training exited with {code}; last report {result.get('stop_reason')}")
    keys = ("steps", "tokens", "stop_reason", "elapsed_seconds", "final_checkpoint", "export")
    return {k: result[k] for k in keys if k in result} or result


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=86400,
    retries=0,
    secrets=[secret],
    volumes=VOLUMES,
)
def train_segment(run_id, config, timeout, deadline, max_tokens=0, gpus=8, provenance=None):
    extra = ["--max-tokens", str(max_tokens)] if max_tokens else []
    return run_segment(
        run_id, config, timeout, deadline, gpus, provenance, "model_hybrid.train", extra
    )


@app.function(
    image=rl_image,
    gpu="H100:8",
    cpu=64,
    memory=262144,
    timeout=86400,
    retries=0,
    secrets=[secret],
    volumes=VOLUMES,
)
def rl_segment(run_id, config, timeout, deadline, gpus=8, provenance=None):
    return run_segment(
        run_id,
        config,
        timeout,
        deadline,
        gpus,
        provenance,
        "model_hybrid.rl",
        extendable=("steps",),
    )


@app.function(
    image=image, gpu="H100:8", cpu=32, memory=262144, timeout=1800, retries=0, volumes=VOLUMES
)
def benchmark(
    config, gpus, steps, microbatch, global_rows, document_tokens, deadline, profile=False
):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from model_hybrid.benchmark import summarize

    path = Path("/tmp/run-config.json")
    path.write_text(json.dumps(config))
    args = ["--config", str(path), "--steps", str(steps), "--document-tokens", str(document_tokens)]
    if microbatch:
        args += ["--microbatch", str(microbatch)]
    if global_rows:
        args += ["--global-rows", str(global_rows)]
    if profile:
        args += ["--profile", "/tmp/profile.json"]
    code, lines = run_torchrun("model_hybrid.benchmark", args, gpus, 1500, Path("/tmp/bench.log"))
    if code:
        raise RuntimeError("".join(lines[-60:]))
    import torch

    return {
        "gpus": gpus,
        "microbatch": microbatch or config["microbatch"],
        "torch": torch.__version__,
        **summarize(lines),
        **({"profile": json.loads(Path("/tmp/profile.json").read_text())} if profile else {}),
    }


@app.function(image=image, cpu=0.25, memory=512, timeout=86400, retries=0, volumes=VOLUMES)
def enforce_deadline(run_id, deadline):
    """Independent supervisor: cancel every recorded call of this launch at its deadline.

    Calls are recorded per launch (keyed by its deadline), so a later launch of the
    same run is never cancelled by an earlier launch's supervisor.
    """
    from workflows.deadline import calls_finished, cancel_at_deadline, tracked_calls

    folder = Path("/runs/.calls") / run_id / launch_key(deadline)

    def calls():
        runs_volume.reload()
        return tracked_calls(folder)

    # The launch is over once its recorded calls have all finished.
    cancelled = cancel_at_deadline(deadline, calls, done=lambda: calls_finished(calls()))
    return {"stage": "completed"} if cancelled is None else {"cancelled": cancelled}


def launch_key(deadline):
    return str(int(deadline))


def load_config(path):
    return json.loads(Path(path).read_text())


@app.local_entrypoint()
def bench(
    config: str = DEFAULT_CONFIG,
    gpus: int = 1,
    steps: int = 20,
    microbatch: int = 0,
    global_rows: int = 0,
    document_tokens: int = 1000,
    profile: bool = False,
    mtp_loss_fraction: float = 0.0,
):
    """One bounded throughput benchmark; with one GPU the global batch shrinks to fit.

    `--profile` adds GPU kernel time per category for a few steps (see benchmark.py);
    `--mtp-loss-fraction` overrides the model's MTP loss fraction.
    """
    from workflows.deadline import MAX_LAUNCH_SECONDS

    settings = load_config(config)
    if mtp_loss_fraction:
        settings["model"]["mtp_loss_fraction"] = mtp_loss_fraction
    rows = global_rows or (
        settings["global_rows"] if gpus == 8 else 8 * (microbatch or settings["microbatch"]) * gpus
    )
    result = benchmark.with_options(gpu=f"H100:{gpus}").remote(
        settings,
        gpus,
        steps,
        microbatch,
        rows,
        document_tokens,
        time.time() + MAX_LAUNCH_SECONDS,
        profile,
    )
    print(json.dumps(result, indent=2))


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES
)
def heldout_loss(export, recipe, counts, deadline):
    """Held-out loss and bits per byte of an export on a pretraining recipe's sources."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_hybrid import generate
    from model_hybrid.train import evaluate, heldout_windows

    lm = generate.load_export(Path("/runs") / export, device="cuda")
    root = Path("/data") / recipe["name"]
    windows = {
        name: heldout_windows(recipe, root, name, lm.config.block_size, count)
        for name, count in counts.items()
    }
    torch.backends.cuda.matmul.allow_tf32 = True
    results = evaluate(lm, windows, torch.device("cuda"), lm.config.eos_id)
    return {
        name: {"loss": r["loss"], "bits_per_byte": r["bits_per_byte"]}
        for name, r in results.items()
    }


@app.local_entrypoint()
def heldout(export: str, recipe: str = "hybrid/recipes/hybrid-40b-v1.json"):
    """Pretraining held-out loss of `<run id>/export/<export>` (the 40B run's document counts)."""
    from workflows.deadline import MAX_LAUNCH_SECONDS

    counts = {"climbmix": 1000, "code": 300, "math": 300}
    result = heldout_loss.remote(
        export, load_config(recipe), counts, time.time() + MAX_LAUNCH_SECONDS
    )
    print(json.dumps({"export": export, "heldout": result}, indent=1))


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def export_checkpoint(run_id):
    """bf16 trunk export (config.json, model.safetensors) of a run's latest checkpoint."""
    import dataclasses

    import torch
    from model_hybrid import tokenizer as glm
    from safetensors.torch import save_file

    from trainer.loop import latest_checkpoint

    folder = Path("/runs") / run_id
    latest = latest_checkpoint(folder)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    settings = json.loads((folder / "run-config.json").read_text())["model"]
    from model_hybrid.model import HybridConfig

    model_config = HybridConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()}
    )
    out = folder / "export" / latest.stem
    out.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.to(torch.bfloat16).contiguous()
        for k, v in state["model"].items()
        if not k.startswith("mtp.")
    }
    save_file(weights, out / "model.safetensors")
    config = dataclasses.asdict(model_config) | {
        "tokenizer": glm.SOURCE,
        "tokenizer_sha256": glm.SHA256,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    progress = {"step": state["step"], "tokens": state["tokens"]}
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    runs_volume.commit()
    return {"export": f"{run_id}/{latest.stem}", "progress": progress}


@app.local_entrypoint()
def export(run_id: str):
    """Export a run's latest checkpoint for evaluation (`<run id>/export/step-N/`)."""
    print(json.dumps(export_checkpoint.remote(run_id)))


@app.local_entrypoint()
def stop(run_id: str):
    """Ask the run's segment to checkpoint and exit at its next step."""
    control[run_id] = "stop"
    print(json.dumps({"run_id": run_id, "requested": "stop"}))


def start(run_id, config, segment, kind, gpus, extra=(), notes=None):
    """Fix the deadline, start the supervisor, spawn and record one segment."""
    import re

    from workflows.deadline import bounded_timeout, new_deadline

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    settings = load_config(config)
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "config_file": config,
        **(notes or {}),
    }
    control[run_id] = "run"
    deadline = new_deadline()
    supervisor = enforce_deadline.spawn(run_id, deadline)
    timeout = bounded_timeout(deadline, 86400)
    call = segment.with_options(timeout=timeout, gpu=f"H100:{gpus}").spawn(
        run_id, settings, timeout, deadline, *extra, gpus=gpus, provenance=provenance
    )
    # Same record as workflows.deadline.track, which the supervisor reads.
    record = {"call_id": call.object_id, "kind": kind, "spawned_unix": time.time()}
    with runs_volume.batch_upload(force=True) as upload:
        upload.put_file(
            io.BytesIO((json.dumps(record) + "\n").encode()),
            f".calls/{run_id}/{launch_key(deadline)}/calls/{call.object_id}.json",
        )
    print(
        json.dumps(
            {
                "deadline_unix": deadline,
                "supervisor": supervisor.object_id,
                "segment": call.object_id,
                "timeout": timeout,
            }
        )
    )


@app.local_entrypoint()
def launch(run_id: str, config: str = DEFAULT_CONFIG, max_tokens: int = 0, gpus: int = 8):
    start(
        run_id,
        config,
        train_segment,
        "train_segment",
        gpus,
        (max_tokens,),
        {"max_tokens": max_tokens},
    )


@app.local_entrypoint()
def launch_rl(run_id: str, config: str, gpus: int = 8):
    """An RL run (model_hybrid.rl) with the same deadline supervision as training."""
    start(run_id, config, rl_segment, "rl_segment", gpus)
