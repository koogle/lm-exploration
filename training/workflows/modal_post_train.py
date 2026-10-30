"""The post-training stages' GPU workers (`workflows/modal_train.py` runs workflows).
Importing this module never starts a job.

Each post-training stage has a worker (`WORKERS`, by stage, with its time limit) called as
`worker(recipe, examples, deadline, launch)`, where `recipe` is the stage's validated
inputs with its run ID and parent (`workflows.stages.validate`) and `examples` any local
SFT data; it returns the run's result with its final `checkpoint`. Training stops taking
updates 30 minutes before the launch deadline to evaluate and save. No call retries.

- SFT: one training call on the stage's GPU (`trainer/sft.py`). `qualify_sft` checks
  exact resume first: an uninterrupted run against stop, fresh-container restore and
  continue.
- RL: one H100 training call (`trainer/rl/`) that runs model-written code only in a
  network-blocked sandbox, recorded for the supervisor. The workflow's family picks the
  worker: subword runs on the subword image with its runs on the training volume; hybrid
  on the hybrid image (its kernels) with parents and runs on the hybrid runs volume and
  the prompt pool on the training volume.
"""

import json
import time
from pathlib import Path

from model_hybrid.image import build as hybrid_image_with

from distributed.image import project_image
from workflows.modal_app import (
    app,
    control,
    data_volume,
    hybrid_runs,
    run_volume,
    track_call,
)

# Seconds a call may run: SFT's settings cap its training loop at four hours, RL's at 20.
SFT_SECONDS = 6 * 60 * 60
RL_SECONDS = 21 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
LOCAL_DATA_LIMIT = 20 * 1024 * 1024

image = project_image("math-verify==0.8.0")
hybrid_image = hybrid_image_with(("math-verify==0.8.0",))
volumes = {"/runs": run_volume, "/data": data_volume}
# Hybrid post-training: parents and runs on the hybrid runs volume, pools beside.
hybrid_volumes = {"/runs": hybrid_runs, "/pools": run_volume}


def train_sft(recipe, examples, deadline, stop_after_step=None):
    from training_data.subword import digest

    from trainer.sft import Settings, train
    from workflows.stages import validate

    if time.time() >= deadline - FINAL_RESERVE_SECONDS:
        raise TimeoutError("Launch deadline passed; refusing SFT work")
    recipe = validate(recipe)
    run_volume.reload()
    runs = Path("/runs")
    if "volume" in recipe["data"]:
        folder = runs / recipe["data"]["volume"]
        report = json.loads((folder / "report.json").read_text())
        dataset = folder / "examples.jsonl"
        if digest(dataset) != report["dataset_sha256"]:
            raise ValueError("Dataset differs from its preparation report")
    else:
        if not examples or len(examples) > LOCAL_DATA_LIMIT:
            raise ValueError("Local SFT data must be nonempty and at most 20 MiB")
        dataset = Path("/tmp/sft-examples.jsonl")
        dataset.write_bytes(examples)
    try:
        result = train(
            runs / recipe["parent"],
            dataset,
            runs / recipe["run_id"],
            Settings(**recipe["settings"]),
            run_volume.commit,
            resume=runs / recipe["resume"] if recipe["resume"] else None,
            extend=recipe["extend"],
            stop_after_step=stop_after_step,
            deadline=deadline - FINAL_RESERVE_SECONDS,
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        run_volume.commit()


@app.function(
    image=image, gpu="L40S", cpu=4, memory=16384, timeout=SFT_SECONDS, retries=0, volumes=volumes
)
def sft_l40s(recipe, examples, deadline, launch, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=SFT_SECONDS, retries=0, volumes=volumes
)
def sft_h100(recipe, examples, deadline, launch, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


def train_rl(recipe, deadline, launch, runs_volume, pools):
    """One RL run: parent and output under /runs (`runs_volume`), pool and trajectory
    data under `pools`."""
    from training_data.sandbox import Sandbox
    from training_data.subword import digest

    from trainer.rl import Settings
    from trainer.rl.loop import train
    from workflows.stages import validate

    stop = deadline - FINAL_RESERVE_SECONDS
    if time.time() >= stop:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    recipe = validate(recipe)
    for volume in {runs_volume, run_volume}:
        volume.reload()
    runs, pools = Path("/runs"), Path(pools)
    pool = pools / recipe["pool"]
    if digest(pool / "pool.jsonl") != json.loads((pool / "report.json").read_text())["pool_sha256"]:
        raise ValueError("Pool differs from its preparation report")
    settings = Settings.from_dict(recipe["settings"])
    trajectories = None
    if settings.trajectories.source == "file":
        folder = pools / settings.trajectories.data
        trajectories = folder / "trajectories.jsonl"
        report = json.loads((folder / "report.json").read_text())
        if digest(trajectories) != report["trajectories_sha256"]:
            raise ValueError("Trajectories differ from their report")
    sandbox = Sandbox(app, deadline - time.time())
    try:
        track_call(launch, sandbox.object_id, "sandbox")
        result = train(
            runs / recipe["parent"],
            pool / "pool.jsonl",
            runs / recipe["run_id"],
            settings,
            sandbox,
            runs_volume.commit,
            deadline=stop,
            trajectories_path=trajectories,
            # The workflow's `stop` entrypoint: checkpoint and pause at the next step.
            should_stop=lambda: control.get(launch) == "stop",
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        sandbox.close()
        runs_volume.commit()


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=RL_SECONDS, retries=0, volumes=volumes
)
def rl_h100(recipe, examples, deadline, launch):
    return train_rl(recipe, deadline, launch, run_volume, "/runs")


@app.function(
    image=hybrid_image,
    gpu="H100",
    cpu=8,
    memory=131072,
    timeout=RL_SECONDS,
    retries=0,
    volumes=hybrid_volumes,
)
def rl_h100_hybrid(recipe, examples, deadline, launch):
    return train_rl(recipe, deadline, launch, hybrid_runs, "/pools")


# {stage: (the worker for a recipe and family, seconds it may run)}
WORKERS = {
    "sft": (
        lambda recipe, family: {"L40S": sft_l40s, "H100": sft_h100}[recipe["gpu"]],
        SFT_SECONDS,
    ),
    "rl": (
        lambda recipe, family: {"subword": rl_h100, "hybrid": rl_h100_hybrid}[family],
        RL_SECONDS,
    ),
}


@app.function(
    image=image, cpu=2, memory=12288, timeout=600, retries=0, volumes={"/runs": run_volume}
)
def compare_sft(left, right, deadline):
    """Exact equality of two SFT checkpoints (uninterrupted versus restored)."""
    import torch

    from workflows.checks import checkpoint

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing comparison")
    run_volume.reload()
    a = torch.load(Path("/runs") / checkpoint(left), map_location="cpu", weights_only=False)
    b = torch.load(Path("/runs") / checkpoint(right), map_location="cpu", weights_only=False)

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)):
            return len(x) == len(y) and all(equal(v, w) for v, w in zip(x, y, strict=True))
        return x == y

    keys = (
        "identity",
        "model",
        "optimizer",
        "step",
        "answer_tokens",
        "sampling_rng",
        "replay_state",
        "torch_rng",
        "cuda_rng",
    )
    checks = {key: equal(a[key], b[key]) for key in keys}
    return {"passed": all(checks.values()), "exact": checks, "step": a["step"]}


def start(function, seconds, launch, deadline, *args, **kwargs):
    """Spawn `function` within the launch deadline, recorded for the supervisor; a call
    that cannot be recorded is cancelled."""
    from workflows.deadline import bounded_timeout

    call = function.with_options(timeout=bounded_timeout(deadline, seconds)).spawn(*args, **kwargs)
    try:
        track_call(launch, call.object_id, "post_train")
    except BaseException:
        call.cancel()
        raise
    return call


def run_stage(stage, family, recipe, launch, deadline, examples=b""):
    """Start one post-training stage's training call for the workflow's model family."""
    worker, seconds = WORKERS[stage]
    return start(
        worker(recipe, family), seconds, launch, deadline, recipe, examples, deadline, launch
    )


def relative(path):
    return str(Path(path).relative_to("/runs"))


def qualify_sft(recipe, launch, deadline, steps, examples=b""):
    """Compare an uninterrupted `steps`-step run with stop at `steps - 1`, fresh-container
    restore and continue, as runs `<run ID>-full`, `-part` and `-resume`; the report.

    A stage with `resume` and `extend` qualifies an extension from that checkpoint; set
    `steps` to its restored step plus two."""
    prefix = recipe["run_id"]
    worker, seconds = WORKERS["sft"]
    worker = worker(recipe, "subword")
    recipe = {**recipe, "settings": {**recipe["settings"], "steps": steps}}

    def run(changes, **kwargs):
        changed = {**recipe, **changes}
        return start(
            worker, seconds, launch, deadline, changed, examples, deadline, launch, **kwargs
        ).get()

    full = run({"run_id": prefix + "-full"})
    part = run({"run_id": prefix + "-part"}, stop_after_step=steps - 1)
    resumed = run(
        {"run_id": prefix + "-resume", "resume": relative(part["checkpoint"]), "extend": False}
    )
    report = start(
        compare_sft,
        600,
        launch,
        deadline,
        relative(full["checkpoint"]),
        relative(resumed["checkpoint"]),
        deadline,
    ).get()
    report.update(full=full, partial=part, resumed=resumed)
    if not report["passed"]:
        raise RuntimeError(f"Resume qualification failed: {report['exact']}")
    return report


def local_examples(recipe):
    """A local SFT data file, validated before a GPU is allocated; volume data is checked
    remotely."""
    if "local" not in recipe.get("data", {}):
        return b""
    from training_data.subword import load_frozen

    from trainer.sft import TASKS

    data = TASKS[recipe["settings"]["task"]]
    path = Path(recipe["data"]["local"])
    data.prepare(path, load_frozen(), recipe["settings"]["context"])
    return path.read_bytes()
