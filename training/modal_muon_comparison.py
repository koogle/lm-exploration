"""Fresh matched optimizer comparison; sequential, bounded eight-H100 jobs."""

import json
import os
import subprocess
from pathlib import Path

import modal

from modal_distributed_train import validate_paths, worker
from modal_simple_llm_8k import SNAPSHOT, data_volume, image, run_volume

image = image.add_local_file("modal_simple_llm_8k.py", "/project/modal_simple_llm_8k.py")

app = modal.App("simple-llm-muon-comparison")


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
def run_arm(run_id, settings, provenance):
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    subprocess.run(
        [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "/project/tests",
            "-p",
            "test_muon.py",
            "-v",
        ],
        check=True,
        timeout=120,
    )
    if settings["optimizer"] == "muon":
        subprocess.run(
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "/project/tests/check_muon_cuda.py",
            ],
            check=True,
            timeout=600,
        )
    # Both arms start from the same seed and cursor; never load an old AdamW model.
    first = worker.local(run_id + "-start", SNAPSHOT, settings, provenance, "", 900, 2)
    resume = str(Path(first["final_checkpoint"]).relative_to("/runs"))
    report = worker.local(run_id, SNAPSHOT, settings, provenance, resume, 3600, None)
    if first["final_validation"] != report["initial_validation"]:
        raise RuntimeError("Fresh-process restore changed validation")
    rows = [
        json.loads(s) for s in (Path("/runs") / run_id / "metrics.jsonl").read_text().splitlines()
    ]
    steady = rows[10:]
    if not steady:
        raise RuntimeError("Too few steps for steady throughput")
    return {
        "report": report,
        "initial_validation": first["initial_validation"],
        "resume_validation": "exact",
        "steady_steps": len(steady),
        "nonpadding_tokens_per_second": sum(r["loss_tokens"] for r in steady)
        / sum(r["seconds"] for r in steady),
        "whole_run_seconds": first["elapsed_seconds"] + report["elapsed_seconds"],
        "peak_allocated_gib": max(v["peak_allocated_bytes"] for r in rows for v in r["ranks"])
        / 2**30,
    }


@app.local_entrypoint()
def main(prefix: str):
    validate_paths(prefix + "-adamw-start", SNAPSHOT, "")
    settings = json.loads(Path("recipes/distributed/h100x8-muon-comparison.json").read_text())
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "purpose": "Fresh matched 100M-token optimizer comparison; single seed; diagnostic validation",
    }
    results = {}
    destination = Path("results") / (prefix + ".json")
    if destination.exists():
        raise ValueError("Comparison results already exist")
    for optimizer in ("adamw", "muon"):
        results[optimizer] = run_arm.remote(
            prefix + "-" + optimizer, {**settings, "optimizer": optimizer}, provenance
        )
        destination.write_text(json.dumps(results, indent=2) + "\n")
        report = results[optimizer]["report"]
        if (
            report["stop_reason"] != "token_limit"
            or not 0
            <= settings["target_tokens"] - report["total_loss_tokens"]
            < settings["global_windows"] * settings["context"]
        ):
            raise RuntimeError("Incomplete arm; retained report, stopping comparison")
    a, m = results["adamw"], results["muon"]
    for key in ("total_loss_tokens", "loss_token_exposures", "data_identity"):
        if a["report"][key] != m["report"][key]:
            raise RuntimeError(f"Unmatched comparison: {key}")
    if a["initial_validation"] != m["initial_validation"]:
        raise RuntimeError("Initial validation differs")
    print(json.dumps(results, indent=2))
