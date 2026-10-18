"""Isolated, bounded eight-H100 optimization experiments; never deploys the 5B app."""

import json
import subprocess
from pathlib import Path

import modal

from modal_distributed_train import worker
from modal_simple_llm_8k import SNAPSHOT, data_volume, image, run_volume

image = image.add_local_file("modal_simple_llm_8k.py", "/project/modal_simple_llm_8k.py")

app = modal.App("simple-llm-optimization-benchmarks")


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=65536,
    timeout=5400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def benchmark(run_id, settings, provenance, qualify=False):
    if qualify:
        result = subprocess.run(
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "/project/tests/check_optimization_cuda.py",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        folder = Path("/runs") / (run_id + "-qualification")
        folder.mkdir(exist_ok=False)
        (folder / "cuda.log").write_text(result.stdout + result.stderr)
        run_volume.commit()
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-6000:])
    # Fresh-job restore exercises model/optimizer/RNG/cursor state for every variant.
    first = worker.local(run_id + "-start", SNAPSHOT, settings, provenance, "", 600, 2)
    resume = str(Path(first["final_checkpoint"]).relative_to("/runs"))
    report = worker.local(run_id, SNAPSHOT, settings, provenance, resume, 3600, None)
    if first["final_validation"] != dict(report["initial_validation"]):
        raise RuntimeError("Fresh-process resume validation mismatch")
    rows = [
        json.loads(line)
        for line in (Path("/runs") / run_id / "metrics.jsonl").read_text().splitlines()
    ]
    steady = rows[10:]
    summary = {
        "run_id": run_id,
        "settings": settings,
        "report": report,
        "resume_validation": "exact",
        "measured_steps": len(steady),
        "warmup_steps_excluded": 10,
        "nonpadding_tokens_per_second": sum(r["loss_tokens"] for r in steady)
        / sum(r["seconds"] for r in steady),
        "peak_allocated_bytes": max(v["peak_allocated_bytes"] for r in rows for v in r["ranks"]),
        "peak_reserved_bytes": max(v["peak_reserved_bytes"] for r in rows for v in r["ranks"]),
        "rank_compute_seconds": [
            sum(r["ranks"][i]["compute_seconds"] for r in steady) for i in range(8)
        ],
        "rank_nonpadding_tokens": [
            sum(r["ranks"][i]["nonpadding_tokens"] for r in steady) for i in range(8)
        ],
    }
    (Path("/runs") / run_id / "benchmark.json").write_text(json.dumps(summary, indent=2))
    run_volume.commit()
    return summary


@app.function(
    image=image, cpu=1, memory=2048, timeout=43200, retries=0, volumes={"/runs": run_volume}
)
def suite(prefix, provenance):
    settings = json.loads(Path("/project/settings.json").read_text())
    settings.update(target_tokens=100_000_000, checkpoint_every=10000, evaluate_every=10000)
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
    folder = Path("/runs") / prefix
    folder.mkdir(exist_ok=False)
    results = []
    for index, (name, changes) in enumerate(variants):
        try:
            result = benchmark.remote(
                prefix + "-" + name, {**settings, **changes}, provenance, qualify=index == 0
            )
        except Exception as exc:  # noqa: BLE001 -- retain failed variants in the experiment record
            result = {"variant": name, "status": "failed", "error": str(exc)}
        results.append(result)
        run_volume.reload()
        (folder / "results.json").write_text(json.dumps(results, indent=2))
        run_volume.commit()
        print(json.dumps(result), flush=True)
        if index == 0 and result.get("status") == "failed":
            raise RuntimeError("Baseline/correctness gate failed; no optimization runs started")
    return results


@app.local_entrypoint()
def main(prefix: str):
    from modal_distributed_train import validate_paths

    validate_paths(prefix, SNAPSHOT, "")
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "purpose": "100M nonpadding tokens per variant, fixed global batch 32",
    }
    results = suite.remote(prefix, provenance)
    Path(f"results/{prefix}.json").write_text(json.dumps(results, indent=2) + "\n")
