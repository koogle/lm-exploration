"""Explicit 25x-longer matched continuation of the completed CIG pilot."""

import json
from pathlib import Path

import modal

from modal_cig_pilot import collect, image
from modal_repair_sft import compare, worker

image = image.add_local_file("modal_cig_pilot.py", "/project/modal_cig_pilot.py")
app = modal.App("simple-llm-cig-long")
app.include(collect.app)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=2,
    single_use_containers=True,
    volumes={"/runs": modal.Volume.from_name("lm-exploration-training")},
)
def extend_worker(run_id, examples, settings, resume):
    # Same production worker body, with a two-container cap for the matched pair.
    return worker.local(run_id, examples, settings, resume=resume, extend=True)


@app.local_entrypoint()
def qualify_extension(
    output: str = "artifacts/cig-sft-long-20261020", pilot: str = "artifacts/cig-sft-20261020"
):
    from trainer.repair_sft import Settings

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    examples = (Path(pilot) / "repairs.jsonl").read_bytes()
    parent = "cig-repair-20261020-r01-cig/step-000000200.pt"
    settings = Settings(steps=202, objective="cig", save_every=1000, evaluate_every=200).__dict__
    full = worker.remote("cig-long-qual-full", examples, settings, resume=parent, extend=True)
    part = worker.remote(
        "cig-long-qual-part", examples, settings, resume=parent, extend=True, stop_after_step=201
    )
    resumed = worker.remote(
        "cig-long-qual-resume",
        examples,
        settings,
        resume=str(Path(part["checkpoint"]).relative_to("/runs")),
    )
    report = compare.remote(
        str(Path(full["checkpoint"]).relative_to("/runs")),
        str(Path(resumed["checkpoint"]).relative_to("/runs")),
    )
    report.update(full=full, partial=part, resumed=resumed)
    (destination / "qualification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


@app.local_entrypoint()
def main(
    output: str = "artifacts/cig-sft-long-20261020", pilot: str = "artifacts/cig-sft-20261020"
):
    from training_data.repair_sft import prepare
    from training_data.subword import load_frozen

    from trainer.repair_sft import Settings

    destination = Path(output)
    qualification = json.loads((destination / "qualification.json").read_text())
    if not qualification.get("passed"):
        raise ValueError("GPU extension qualification must pass first")
    examples = (Path(pilot) / "repairs.jsonl").read_bytes()
    dataset = destination / "repairs.jsonl"
    dataset.write_bytes(examples)
    prepare(dataset, load_frozen(), 1024)
    calls, report = {}, {}
    for objective in ("sft", "cig"):
        settings = Settings(
            steps=5000, objective=objective, save_every=1000, evaluate_every=500, max_seconds=3000
        ).__dict__
        run_id = f"cig-repair-long-20261020-r01-{objective}"
        parent = f"cig-repair-20261020-r01-{objective}/step-000000200.pt"
        calls[objective] = extend_worker.spawn(run_id, examples, settings, parent)
        report[objective] = {
            "run_id": run_id,
            "call_id": calls[objective].object_id,
            "settings": settings,
            "resume": parent,
        }
    (destination / "launch.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    results = {}
    for objective, call in calls.items():
        results[objective] = {**call.get(), **collect.remote(report[objective]["run_id"])}
        (destination / "training.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"arm": objective, "step": results[objective]["step"]}), flush=True)
