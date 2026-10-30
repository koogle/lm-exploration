"""Post-training stages and pipelines of the workflow controller (`workflows/modal_train.py`).
Importing this module never starts a job.

A post-training workflow file is a stage file (`workflows/stages/`, e.g. `"stage": "sft"`)
or a pipeline of them (`workflows/pipeline.py`).
As for pretraining, a launch has one deadline 24 hours after it starts (less the
supervisor's margin), and the independent supervisor (`workflows.modal_app.enforce_deadline`)
starts first. A CPU controller call spawns each GPU call with a timeout that ends before
the deadline, records the call (and any sandbox a stage starts) for the supervisor, and
records the launch `completed` or `failed` in `/runs/<run-id>/workflow.json`. The
supervisor then terminates the launch's sandboxes, or at the deadline cancels its calls
and terminates its sandboxes. Training stops taking updates 30 minutes before the
deadline to evaluate and save. No call retries.

- SFT: one training call on the recipe's GPU. With `qualify` N, a resume qualification
  instead: an uninterrupted N-step run, a run stopped after N − 1 steps, its resume in a
  fresh container, and an exact comparison of the two final checkpoints.
- A pipeline runs its steps in order; a step's branches run side by side.

Each stage has a GPU worker (`WORKERS`, by stage, with its time limit) called as
`worker(recipe, examples, deadline, launch)`; it returns the run's result with its final
`checkpoint`.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from distributed.image import project_image
from workflows.modal_app import app, data_volume, enforce_deadline, record, run_volume, track_call

# Seconds an SFT call may run: its settings cap the training loop at four hours.
SFT_SECONDS = 6 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
LOCAL_DATA_LIMIT = 20 * 1024 * 1024

image = project_image()
volumes = {"/runs": run_volume, "/data": data_volume}


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
def sft_l40s(recipe, examples, deadline, launch, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=SFT_SECONDS, retries=0, volumes=volumes
)
def sft_h100(recipe, examples, deadline, launch, stop_after_step=None):
    return train_sft(recipe, examples, deadline, stop_after_step)


# {stage: (the worker for a recipe, seconds it may run)}
WORKERS = {"sft": (lambda recipe: {"L40S": sft_l40s, "H100": sft_h100}[recipe["gpu"]], SFT_SECONDS)}


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


def run_stage(stage, recipe, examples, deadline, launch):
    """Start one stage's training call."""
    worker, seconds = WORKERS[stage]
    return start(worker(recipe), seconds, launch, deadline, recipe, examples, deadline, launch)


def relative(path):
    return str(Path(path).relative_to("/runs"))


def qualify_sft(recipe, examples, deadline, steps):
    """Compare an uninterrupted run with stop-at-(steps-1), fresh-container restore, continue.

    A recipe with `resume` and `extend` qualifies an extension from that checkpoint; set
    `steps` to its restored step plus two.
    """
    prefix = recipe["run_id"]
    worker, seconds = WORKERS["sft"]
    worker = worker(recipe)
    recipe = {**recipe, "settings": {**recipe["settings"], "steps": steps}}

    def run(changes, **kwargs):
        changed = {**recipe, **changes}
        return start(
            worker, seconds, prefix, deadline, changed, examples, deadline, prefix, **kwargs
        ).get()

    full = run({"run_id": prefix + "-full"})
    part = run({"run_id": prefix + "-part"}, stop_after_step=steps - 1)
    resumed = run(
        {"run_id": prefix + "-resume", "resume": relative(part["checkpoint"]), "extend": False}
    )
    report = start(
        compare_sft,
        600,
        prefix,
        deadline,
        relative(full["checkpoint"]),
        relative(resumed["checkpoint"]),
        deadline,
    ).get()
    report.update(full=full, partial=part, resumed=resumed)
    if not report["passed"]:
        raise RuntimeError(f"Resume qualification failed: {report['exact']}")
    return report


@app.function(
    image=image, cpu=1, memory=2048, timeout=86400, retries=0, volumes={"/runs": run_volume}
)
def post_train(stage, recipe, examples, provenance, qualify):
    """Controller of one stage's launch; records it `completed` or `failed`."""
    deadline, launch = provenance["deadline_unix"], recipe["run_id"]
    try:
        if qualify:
            result = qualify_sft(recipe, examples, deadline, qualify)
        else:
            result = run_stage(stage, recipe, examples, deadline, launch).get()
    except BaseException as error:
        record(launch, "failed", error=repr(error), provenance=provenance)
        raise
    record(launch, "completed", result=result, provenance=provenance)
    return result


@app.function(
    image=image, cpu=1, memory=2048, timeout=86400, retries=0, volumes={"/runs": run_volume}
)
def run_pipeline(plan, examples, provenance):
    """Controller of one pipeline launch: runs the stages `pipeline.json` does not record
    as finished; records the launch `completed` or `failed`."""
    from workflows.pipeline import MAX_ATTEMPTS, attempt_run
    from workflows.stages import validate

    deadline, launch = provenance["deadline_unix"], plan["run_id"]
    run_volume.reload()
    state_path = Path("/runs") / launch / "pipeline.json"
    state = {"plan_sha256": provenance["plan_sha256"], "stages": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != provenance["plan_sha256"]:
            raise ValueError("This pipeline run was started with another plan")

    def save():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        run_volume.commit()

    parent = plan["from"]
    try:
        for step in plan["steps"]:
            calls = []
            for entry in step:
                name = entry["recipe"]["run_id"]
                previous = state["stages"].get(name, {})
                if previous.get("status") == "completed":
                    continue
                attempt = previous.get("attempt", 0) + 1
                if attempt > MAX_ATTEMPTS:
                    raise RuntimeError(f"{name} did not finish in {MAX_ATTEMPTS} attempts")
                recipe = {**entry["recipe"], "run_id": attempt_run(name, attempt), "parent": parent}
                recipe = validate(recipe)
                state["stages"][name] = {
                    "status": "running",
                    "attempt": attempt,
                    "run_id": recipe["run_id"],
                    "parent": parent,
                }
                save()
                calls.append(
                    (
                        name,
                        run_stage(
                            entry["stage"], recipe, examples.get(name, b""), deadline, launch
                        ),
                    )
                )
            for name, call in calls:
                result = call.get()
                state["stages"][name].update(
                    status="completed", checkpoint=relative(result["checkpoint"]), result=result
                )
                save()
            if len(step) == 1:
                parent = state["stages"][step[0]["recipe"]["run_id"]]["checkpoint"]
    except BaseException as error:
        for stage in state["stages"].values():
            if stage["status"] == "running":
                stage["status"] = "failed"
        save()
        record(launch, "failed", error=repr(error), provenance=provenance)
        raise
    record(launch, "completed", stages=state["stages"], provenance=provenance)
    return state["stages"]


def local_examples(recipe):
    """Validate local SFT data before allocating a GPU; volume data is checked remotely."""
    if recipe.get("data", {}).get("volume") or "data" not in recipe:
        return b""
    from training_data.subword import load_frozen

    from trainer.sft import TASKS

    data = TASKS[recipe["settings"]["task"]]
    path = Path(recipe["data"]["local"])
    data.prepare(path, load_frozen(), recipe["settings"]["context"])
    return path.read_bytes()


def supervised(run, provenance, controller, *args):
    """Start the supervisor, then run `controller` remotely under the launch deadline."""
    # The supervisor starts first so no work runs unsupervised.
    supervisor = enforce_deadline.spawn(run, provenance["deadline_unix"])
    print(
        json.dumps(
            {"deadline_unix": provenance["deadline_unix"], "supervisor": supervisor.object_id}
        )
    )
    return controller.remote(*args)


def provenance_of(workflow):
    from workflows.deadline import new_deadline

    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workflow": workflow,
        # Every call this launch starts stops by this time.
        "deadline_unix": new_deadline(),
    }


def write_result(kind, run, workflow, result):
    path = Path(f"artifacts/{kind}/{run}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workflow": workflow, "result": result}, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))


def launch(stage, workflow, run="", resume="", qualify=0):
    """Validate one stage's workflow file locally, then run it under one deadline."""
    from workflows.checks import run_id
    from workflows.stages import load

    if stage != "sft" and (resume or qualify):
        raise ValueError(f"A {stage} run neither resumes nor qualifies")
    recipe = load(workflow, run_id=run, **({"resume": resume} if resume else {}))
    if qualify:
        run_id(recipe["run_id"] + "-resume", "run_id")
        if qualify < 2:
            raise ValueError("Qualification needs at least two steps")
    provenance = provenance_of(recipe)
    examples = local_examples(recipe)
    result = supervised(
        recipe["run_id"], provenance, post_train, stage, recipe, examples, provenance, qualify
    )
    write_result(stage, recipe["run_id"], recipe, result)
    return result


def launch_pipeline(workflow, run):
    """Validate a pipeline locally, then run its unfinished stages under one deadline."""
    from workflows.pipeline import load_pipeline

    plan = load_pipeline(workflow, run)
    provenance = provenance_of(plan)
    provenance["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    examples = {
        entry["recipe"]["run_id"]: local_examples(entry["recipe"])
        for step in plan["steps"]
        for entry in step
    }
    result = supervised(run, provenance, run_pipeline, plan, examples, provenance)
    write_result("pipeline", run, plan, result)
    return result
