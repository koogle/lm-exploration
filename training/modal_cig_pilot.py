"""Matched 200-update CIG/ordinary-SFT pilot; explicitly launched, no auto-resume."""

import json
from pathlib import Path

import modal

from modal_repair_sft import image, run_volume, worker

image = image.add_local_file("modal_repair_sft.py", "/project/modal_repair_sft.py")

app = modal.App("simple-llm-cig-pilot")
# Reuse the production worker and its bounds, volume and checkpoint format.
app.include(worker.app)


@app.function(image=image, volumes={"/runs": run_volume}, timeout=120)
def collect(run_id):
    run_volume.reload()
    folder = Path("/runs") / run_id
    result = {
        "manifest": json.loads((folder / "manifest.json").read_text()),
        "metrics": [
            json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
        ],
    }
    if (folder / "cig-summary.json").exists():
        result["cig"] = json.loads((folder / "cig-summary.json").read_text())
    if (folder / "continuation.json").exists():
        result["continuation"] = json.loads((folder / "continuation.json").read_text())
    return result


@app.local_entrypoint()
def main(
    output: str = "results/cig-sft-20261020",
    run_prefix: str = "cig-repair-20261020-r01",
    completed_sft: str = "",
):
    from training_data.repair_curriculum import LANGUAGES, rows, write_dataset
    from training_data.repair_sft import prepare
    from training_data.subword import load_frozen

    from trainer.repair_sft import Settings

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "repairs.jsonl"
    write_dataset(dataset, [row for lang in LANGUAGES for row in rows(lang)])
    _, data_hash = prepare(dataset, load_frozen(), 1024)
    report = {}
    for objective in ("sft", "cig"):
        run_id = f"{run_prefix}-{objective}"
        settings = Settings(objective=objective, save_every=200).__dict__
        if objective == "sft" and completed_sft:
            from modal_repair_sft import checkpoint_path

            path = checkpoint_path(completed_sft)
            run_id = path.parent.name
            previous = collect.remote(run_id)
            last = previous["metrics"][-1]
            if (
                last["step"] != settings["steps"]
                or previous["manifest"]["settings"] != settings
                or previous["manifest"]["dataset_sha256"] != data_hash
            ):
                raise ValueError("Completed control does not match settings")
            result = {
                "checkpoint": str(path),
                "step": last["step"],
                "answer_tokens": last["answer_tokens"],
            }
        else:
            result = worker.remote(run_id, dataset.read_bytes(), settings)
        report[objective] = {**result, **collect.remote(run_id)}
        (destination / "training.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"arm": objective, **result}), flush=True)
