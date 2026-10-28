"""Bounded RL (`trainer/rl/`) on one Modal H100. Importing this module never starts a job.

Run from training/:
    modal run -m modal_scripts.modal_rl_train --recipe <recipe.json> [--run-id <fresh-id>] \
        [--output <result.json>]

A recipe names `run_id`, `parent` (an SFT checkpoint, `run/step-N.pt` on the run volume),
`pool` (an `rl_pool` data ID), and `settings` (`trainer/rl/settings.py`; print a
recipe's resolved settings with `python -m trainer.rl --recipe <recipe.json>`).

The launcher records a deadline 24 hours after launch and spawns one training call (no
retries, 21-hour timeout) and one supervisor call. The training call refuses work after
the deadline, stops taking updates 30 minutes before it, and runs model-written code only
in a network-blocked sandbox whose own timeout ends at the deadline. Its sandbox ID is
recorded in `/runs/rl-launches/<run-id>.json`. The supervisor cancels the training call and
terminates the sandbox at the deadline, or terminates the sandbox once the call ends.
"""

import json
import time
from pathlib import Path

import modal

from distributed.image import project_image

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
TIMEOUT_SECONDS = 21 * 60 * 60
FIELDS = {"name", "description", "run_id", "parent", "pool", "settings"}

app = modal.App("simple-llm-rl")
image = project_image("math-verify==0.8.0").add_local_dir(
    "kernels", "/project/kernels", ignore=["__pycache__"]
)
run_volume = modal.Volume.from_name("lm-exploration-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")


def validate_recipe(recipe):
    from trainer.rl import Settings
    from trainer.sft_recipe import checkpoint, run_id

    if not isinstance(recipe, dict) or set(recipe) - FIELDS:
        raise ValueError(f"Recipe fields must be among {sorted(FIELDS)}")
    Settings.from_dict(recipe["settings"])
    run_id(recipe["run_id"])
    run_id(recipe["pool"], "pool")
    checkpoint(recipe["parent"], "parent")
    return recipe


def launch_record(run):
    return Path("/runs/rl-launches") / f"{run}.json"


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=2,
    volumes={"/runs": run_volume, "/data": data_volume},
)
def train(recipe, deadline):
    from training_data.code_execution import Sandbox

    from trainer.rl import Settings
    from trainer.rl.loop import train as run

    stop = deadline - FINAL_RESERVE_SECONDS
    if time.time() >= stop:
        raise TimeoutError("Launch deadline passed; refusing RL work")
    recipe = validate_recipe(recipe)
    run_volume.reload()
    runs = Path("/runs")
    pool = runs / recipe["pool"]
    report = json.loads((pool / "report.json").read_text())
    from training_data.subword import digest

    if digest(pool / "pool.jsonl") != report["pool_sha256"]:
        raise ValueError("Pool differs from its preparation report")
    sandbox = Sandbox(app, deadline - time.time())
    try:
        record = launch_record(recipe["run_id"])
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"deadline": deadline, "sandbox_id": sandbox.object_id}))
        run_volume.commit()
        result = run(
            runs / recipe["parent"],
            pool / "pool.jsonl",
            runs / recipe["run_id"],
            Settings.from_dict(recipe["settings"]),
            sandbox,
            run_volume.commit,
            deadline=stop,
        )
        (runs / recipe["run_id"] / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        sandbox.close()
        run_volume.commit()


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    volumes={"/runs": run_volume},
)
def supervise(call_id, run, deadline):
    """Cancel the training call at the deadline; terminate its sandbox when it ends."""
    call = modal.FunctionCall.from_id(call_id)
    while time.time() < deadline:
        try:
            call.get(timeout=60)
            break
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except Exception:  # noqa: BLE001 - a failed call has ended too
            break
    else:
        call.cancel()
    run_volume.reload()
    record = launch_record(run)
    if record.exists():
        modal.Sandbox.from_id(json.loads(record.read_text())["sandbox_id"]).terminate()


@app.local_entrypoint()
def main(recipe: str, run_id: str = "", output: str = ""):
    spec = json.loads(Path(recipe).read_text())
    if run_id:
        spec["run_id"] = run_id
    spec = validate_recipe(spec)
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = train.spawn(spec, deadline)
    supervise.spawn(call.object_id, spec["run_id"], deadline)
    print(json.dumps({"run_id": spec["run_id"], "call_id": call.object_id}), flush=True)
    result = call.get()
    path = Path(output or f"artifacts/rl/{spec['run_id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recipe": spec, "result": result}, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))
