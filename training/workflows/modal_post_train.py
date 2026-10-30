"""SFT and RL stages of the workflow controller (`workflows/modal_train.py`). Importing this
module never starts a job.

A post-training workflow file is a stage recipe: `"stage": "sft"` (`trainer/sft_recipe.py`)
or `"stage": "rl"` (`trainer/rl/recipe.py`). As for pretraining, a launch has one deadline
24 hours after it starts (less the supervisor's margin), and the independent supervisor
(`workflows.modal_app.enforce_deadline`) starts first. A CPU controller call spawns each
GPU call with a timeout that ends before the deadline, records the call (and the RL code
sandbox) for the supervisor, and records the launch `completed` or `failed` in
`/runs/<run-id>/workflow.json`. The supervisor then terminates the launch's sandboxes, or
at the deadline cancels its calls and terminates its sandboxes. Training stops taking
updates 30 minutes before the deadline to evaluate and save. No call retries.

- SFT: one training call on the recipe's GPU. With `qualify` N, a resume qualification
  instead: an uninterrupted N-step run, a run stopped after N − 1 steps, its resume in a
  fresh container, and an exact comparison of the two final checkpoints.
- RL: one H100 training call that runs model-written code only in a network-blocked
  sandbox.
"""

import json
import subprocess
import time
from pathlib import Path

from distributed.image import project_image
from workflows.modal_app import app, data_volume, enforce_deadline, record, run_volume, track_call

# Seconds each call may run: SFT's settings cap its loop at four hours, RL's at twenty.
SFT_SECONDS = 6 * 60 * 60
RL_SECONDS = 21 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
LOCAL_DATA_LIMIT = 20 * 1024 * 1024

# RL rollouts use the custom decode kernels (`kernels/decode.py`).
image = project_image("math-verify==0.8.0").add_local_dir(
    "kernels", "/project/kernels", ignore=["__pycache__"]
)
volumes = {"/runs": run_volume, "/data": data_volume}


def train_sft(recipe, examples, deadline, stop_after_step=None):
    from training_data.subword import digest

    from trainer.sft import Settings, train
    from trainer.sft_recipe import validate_recipe

    if time.time() >= deadline - FINAL_RESERVE_SECONDS:
        raise TimeoutError("Launch deadline passed; refusing SFT work")
    recipe = validate_recipe(recipe)
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
            stop_after_step=stop_after_step,
            extend=recipe["extend"],
            deadline=deadline - FINAL_RESERVE_SECONDS,
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        run_volume.commit()


@app.function(
    image=image, gpu="L40S", cpu=4, memory=16384, timeout=SFT_SECONDS, retries=0, volumes=volumes
)
def sft_l40s(recipe, examples, deadline, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=SFT_SECONDS, retries=0, volumes=volumes
)
def sft_h100(recipe, examples, deadline, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


SFT_WORKERS = {"L40S": sft_l40s, "H100": sft_h100}


@app.function(
    image=image, cpu=2, memory=12288, timeout=600, retries=0, volumes={"/runs": run_volume}
)
def compare_sft(left, right, deadline):
    """Exact equality of two SFT checkpoints (uninterrupted versus restored)."""
    import torch

    from trainer.sft_recipe import checkpoint

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


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=RL_SECONDS, retries=0, volumes=volumes
)
def rl_h100(recipe, deadline, launch):
    from training_data.sandbox import Sandbox
    from training_data.subword import digest

    from trainer.rl import Settings
    from trainer.rl.loop import train
    from trainer.rl.recipe import validate_recipe

    stop = deadline - FINAL_RESERVE_SECONDS
    if time.time() >= stop:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    recipe = validate_recipe(recipe)
    run_volume.reload()
    runs = Path("/runs")
    pool = runs / recipe["pool"]
    if digest(pool / "pool.jsonl") != json.loads((pool / "report.json").read_text())["pool_sha256"]:
        raise ValueError("Pool differs from its preparation report")
    settings = Settings.from_dict(recipe["settings"])
    trajectories = None
    if settings.trajectories.source == "file":
        folder = runs / settings.trajectories.data
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
            run_volume.commit,
            deadline=stop,
            trajectories_path=trajectories,
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        sandbox.close()
        run_volume.commit()


def spawned(function, seconds, launch, deadline, *args, **kwargs):
    """Run `function` within the launch deadline, recorded for the supervisor; a call that
    cannot be recorded is cancelled."""
    from workflows.deadline import bounded_timeout

    call = function.with_options(timeout=bounded_timeout(deadline, seconds)).spawn(*args, **kwargs)
    try:
        track_call(launch, call.object_id, "post_train")
    except BaseException:
        call.cancel()
        raise
    return call.get()


def qualify_sft(recipe, examples, deadline, steps):
    """Compare an uninterrupted run with stop-at-(steps-1), fresh-container restore, continue.

    A recipe with `resume` and `extend` qualifies an extension from that checkpoint; set
    `steps` to its restored step plus two.
    """
    prefix, worker = recipe["run_id"], SFT_WORKERS[recipe["gpu"]]
    recipe = {**recipe, "settings": {**recipe["settings"], "steps": steps}}
    full = spawned(
        worker,
        SFT_SECONDS,
        prefix,
        deadline,
        {**recipe, "run_id": prefix + "-full"},
        examples,
        deadline,
    )
    part = spawned(
        worker,
        SFT_SECONDS,
        prefix,
        deadline,
        {**recipe, "run_id": prefix + "-part"},
        examples,
        deadline,
        stop_after_step=steps - 1,
    )
    resumed = spawned(
        worker,
        SFT_SECONDS,
        prefix,
        deadline,
        {
            **recipe,
            "run_id": prefix + "-resume",
            "resume": str(Path(part["checkpoint"]).relative_to("/runs")),
            "extend": False,
        },
        examples,
        deadline,
    )
    report = spawned(
        compare_sft,
        600,
        prefix,
        deadline,
        str(Path(full["checkpoint"]).relative_to("/runs")),
        str(Path(resumed["checkpoint"]).relative_to("/runs")),
        deadline,
    )
    report.update(full=full, partial=part, resumed=resumed)
    if not report["passed"]:
        raise RuntimeError(f"Resume qualification failed: {report['exact']}")
    return report


@app.function(
    image=image, cpu=1, memory=2048, timeout=86400, retries=0, volumes={"/runs": run_volume}
)
def post_train(stage, recipe, examples, provenance, qualify):
    """Controller of one SFT or RL launch; records it `completed` or `failed`."""
    deadline, launch = provenance["deadline_unix"], recipe["run_id"]
    try:
        if stage == "rl":
            result = spawned(rl_h100, RL_SECONDS, launch, deadline, recipe, deadline, launch)
        elif qualify:
            result = qualify_sft(recipe, examples, deadline, qualify)
        else:
            worker = SFT_WORKERS[recipe["gpu"]]
            result = spawned(worker, SFT_SECONDS, launch, deadline, recipe, examples, deadline)
    except BaseException as error:
        record(launch, "failed", error=repr(error), provenance=provenance)
        raise
    record(launch, "completed", result=result, provenance=provenance)
    return result


def local_examples(recipe):
    """Validate local SFT data before allocating a GPU; volume data is checked remotely."""
    if "volume" in recipe["data"]:
        return b""
    from training_data.subword import load_frozen

    from trainer.sft import TASKS

    data = TASKS[recipe["settings"]["task"]]
    path = Path(recipe["data"]["local"])
    data.prepare(path, load_frozen(), recipe["settings"]["context"])
    return path.read_bytes()


def launch(stage, workflow, run="", resume="", qualify=0):
    """Validate a post-training workflow file locally, then run it under one deadline."""
    from workflows.deadline import new_deadline

    if stage == "sft":
        from trainer.sft_recipe import load_recipe, run_id

        recipe = load_recipe(workflow, run, resume)
        if qualify:
            run_id(recipe["run_id"] + "-resume", "run_id")
            if qualify < 2:
                raise ValueError("Qualification needs at least two steps")
        examples = local_examples(recipe)
    else:
        from trainer.rl.recipe import load_recipe

        if resume or qualify:
            raise ValueError("An RL run neither resumes nor qualifies")
        recipe, examples = load_recipe(workflow, run), b""
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workflow": recipe,
        # Every call this launch starts stops by this time.
        "deadline_unix": new_deadline(),
    }
    # The supervisor starts first so no work runs unsupervised.
    supervisor = enforce_deadline.spawn(recipe["run_id"], provenance["deadline_unix"])
    print(
        json.dumps(
            {"deadline_unix": provenance["deadline_unix"], "supervisor": supervisor.object_id}
        )
    )
    result = post_train.remote(stage, recipe, examples, provenance, qualify)
    path = Path(f"artifacts/{stage}/{recipe['run_id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workflow": recipe, "result": result}, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))
    return result
