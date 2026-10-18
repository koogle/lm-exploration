"""Single-H100 optimization suite, independent of the queued eight-GPU app."""

import json
from pathlib import Path

import modal

from modal_training_optimizations import benchmark, data_volume, image, run_volume

image = image.add_local_file(
    "modal_training_optimizations.py", "/project/modal_training_optimizations.py"
)
app = modal.App("simple-llm-single-gpu-optimizations")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def trial(run_id, settings, provenance, qualify=False):
    if settings["world_size"] != 1:
        raise ValueError("Expected one GPU")
    return benchmark.local(run_id, settings, provenance, qualify)


@app.function(
    image=image, cpu=1, memory=2048, timeout=43200, retries=0, volumes={"/runs": run_volume}
)
def run_suite(prefix, provenance):
    variants = [
        ("baseline-m1", {}),
        ("crop-m1", {"crop_padding": True}),
        ("crop-m2", {"crop_padding": True, "microbatch": 2}),
        ("crop-m4", {"crop_padding": True, "microbatch": 4}),
        ("chunk-m2", {"crop_padding": True, "microbatch": 2, "loss_chunk_size": 1024}),
        ("rope-m2", {"crop_padding": True, "microbatch": 2, "cache_rope": True}),
        (
            "compile-m2",
            {"crop_padding": True, "microbatch": 2, "cache_rope": True, "compile_model": True},
        ),
    ]
    return run_variants(prefix, provenance, variants)


@app.function(
    image=image, cpu=1, memory=2048, timeout=18000, retries=0, volumes={"/runs": run_volume}
)
def run_m4_fusion_suite(prefix, provenance):
    common = {"crop_padding": True, "microbatch": 4, "cache_rope": True}
    return run_variants(
        prefix,
        provenance,
        [
            ("rope-m4", common),
            ("compile-m4", {**common, "compile_model": True}),
        ],
    )


def run_variants(prefix, provenance, variants):
    settings = json.loads(Path("/project/settings.json").read_text())
    settings.update(
        world_size=1, target_tokens=100_000_000, checkpoint_every=10000, evaluate_every=10000
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
