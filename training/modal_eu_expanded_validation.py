"""Larger matched EU continuation validation; two H100s, no training or HF writes."""

import json
from pathlib import Path

import modal

from modal_eu_continuation import PARENT, data_volume, image, run_volume

image = image.add_local_file("modal_eu_continuation.py", "/project/modal_eu_continuation.py")
app = modal.App("simple-llm-eu-expanded-validation")
ORIGINAL = "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017/manifest.json"
EU = "/data/scholarly-code/eu-continuation-20261018/snapshots/ecb-ema-one-pass-v1/manifest.json"
FINAL = "simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt"


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def prepare(run_id):
    from trainer.eu_expanded_validation import build_sample

    return build_sample(ORIGINAL, EU, Path("/runs") / run_id, run_volume.commit)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=24576,
    timeout=5400,
    retries=0,
    max_containers=2,
    volumes={"/runs": run_volume},
)
def score(run_id, checkpoint, label):
    from trainer.eu_expanded_validation import score_checkpoint

    folder = Path("/runs") / run_id
    result = score_checkpoint(
        Path("/runs") / checkpoint, folder, folder / f"scores-{label}.json", run_volume.commit
    )
    return {
        "completed": result["completed"],
        "total_tokens": result["total_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


@app.function(
    image=image, cpu=1, memory=4096, timeout=14400, retries=0, volumes={"/runs": run_volume}
)
def start(run_id, provenance):
    from distributed_training.train import atomic_json
    from trainer.eu_expanded_validation import paired_summary

    folder = Path("/runs") / run_id
    folder.mkdir(exist_ok=False)
    atomic_json(folder / "provenance.json", provenance)
    atomic_json(folder / "status.json", {"stage": "preparing"})
    run_volume.commit()
    try:
        sample = prepare.remote(run_id)
        calls = [score.spawn(run_id, PARENT, "before"), score.spawn(run_id, FINAL, "after")]
        run_volume.reload()
        atomic_json(
            folder / "status.json",
            {
                "stage": "evaluating",
                "sample": sample,
                "call_ids": [call.object_id for call in calls],
            },
        )
        run_volume.commit()
        timing = [call.get() for call in calls]
        run_volume.reload()
        before = json.loads((folder / "scores-before.json").read_text())
        after = json.loads((folder / "scores-after.json").read_text())
        result = paired_summary(before, after, sample)
        result["timing"] = timing
        atomic_json(folder / "comparison.json", result)
        atomic_json(
            folder / "status.json",
            {"stage": "completed", "total_tokens_per_checkpoint": sample["total_tokens"]},
        )
        run_volume.commit()
        return result
    except Exception as exc:
        run_volume.reload()
        atomic_json(folder / "status.json", {"stage": "failed", "error": str(exc)[:1000]})
        run_volume.commit()
        raise


@app.function(image=image, cpu=1, memory=4096, timeout=120, volumes={"/runs": run_volume})
def inspect(run_id):
    folder = Path("/runs") / run_id
    return {
        name: json.loads((folder / name).read_text())
        for name in ("status.json", "sample-summary.json", "comparison.json")
        if (folder / name).exists()
    }
