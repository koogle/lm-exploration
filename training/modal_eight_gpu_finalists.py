"""Matched eight-H100 comparison of the three selected optimization configurations."""

import json
import os
from pathlib import Path

import modal

from modal_training_optimizations import benchmark, data_volume, image, run_volume

image = image.add_local_file(
    "modal_training_optimizations.py", "/project/modal_training_optimizations.py"
)
app = modal.App("simple-llm-eight-gpu-finalists")


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=65536,
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def trial(run_id, settings, provenance, qualify=False):
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    if settings["world_size"] != 8:
        raise ValueError("Expected eight GPUs")
    return benchmark.local(run_id, settings, provenance, qualify)


@app.function(
    image=image, cpu=1, memory=2048, timeout=43200, retries=0, volumes={"/runs": run_volume}
)
def run_suite(prefix, provenance):
    common = {"crop_padding": True, "cache_rope": True}
    variants = [
        ("rope-m4", {**common, "microbatch": 4}),
        ("compile-m4", {**common, "microbatch": 4, "compile_model": True}),
        ("compile-m2", {**common, "microbatch": 2, "compile_model": True}),
    ]
    return run_variants(prefix, provenance, variants)


def run_variants(prefix, provenance, variants):
    settings = json.loads(Path("/project/settings.json").read_text())
    settings.update(
        world_size=8, target_tokens=100_000_000, checkpoint_every=10000, evaluate_every=10000
    )
    folder = Path("/runs") / prefix
    folder.mkdir(exist_ok=False)
    results = []
    for index, (name, changes) in enumerate(variants):
        try:
            result = trial.remote(
                prefix + "-" + name, {**settings, **changes}, provenance, qualify=index == 0
            )
        except Exception as exc:  # noqa: BLE001 -- preserve failed experiment records
            result = {"variant": name, "status": "failed", "error": str(exc)}
        results.append(result)
        run_volume.reload()
        (folder / "results.json").write_text(json.dumps(results, indent=2))
        run_volume.commit()
        print(json.dumps(result), flush=True)
        if index == 0 and result.get("status") == "failed":
            raise RuntimeError("Baseline/correctness gate failed")
    return results
