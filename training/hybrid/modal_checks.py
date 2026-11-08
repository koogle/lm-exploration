"""Small CUDA correctness checks for the hybrid family on one H100.

Run from training/:
    modal run -m model_hybrid.modal_checks::main [--output <result.json>] [--only hnet|decode]
    modal run -m model_hybrid.modal_checks::kda_bench [--output <result.json>]

One H100, 30-minute timeout, no retries; the call refuses to start once 24 hours
have passed since launch. Results are JSON in the ignored artifacts/results/.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid.image import LAUNCH_LIMIT_SECONDS, build, image

precision_image = build(extra_packages=("torchao==0.18.0",))
app = modal.App("lm-exploration-hybrid-checks")


@app.function(
    image=precision_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=1800,
    retries=0,
    volumes={"/data": modal.Volume.from_name("lm-exploration-hybrid-data")},
)
def check(deadline, only=""):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    if only == "fp8_loss":
        from model_hybrid.fp8_loss_diagnostics import run

        return run()
    if only == "fp8_backward":
        from model_hybrid.float8_diagnostics import run

        return run()
    import runpy

    return runpy.run_path("/project/tests/check_hybrid_cuda.py")["checks"](only)


@app.local_entrypoint()
def main(output: str = "", only: str = ""):
    from workflows.deadline import new_deadline

    deadline = new_deadline()
    call = check.with_options(gpu="H100!").spawn(deadline, only)
    try:
        supervisor = enforce_deadline.spawn([call.object_id], deadline)
    except Exception:
        call.cancel()
        raise
    print(
        json.dumps(
            {"call": call.object_id, "supervisor": supervisor.object_id, "deadline": deadline}
        ),
        flush=True,
    )
    result = call.get()
    path = Path(output or f"artifacts/results/hybrid-cuda-check-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


@app.function(image=image, cpu=0.25, memory=512, timeout=86400, retries=0)
def enforce_deadline(call_ids, deadline):
    from workflows.deadline import calls_finished, cancel_at_deadline

    return cancel_at_deadline(deadline, lambda: call_ids, lambda: calls_finished(call_ids))


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1800, retries=0)
def kda_timing(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from model_hybrid import kda_bench

    return kda_bench.run()


@app.local_entrypoint()
def kda_bench(output: str = ""):
    """KDA kernel timing and accuracy at the 720M shapes (model_hybrid.kda_bench)."""
    result = kda_timing.remote(time.time() + LAUNCH_LIMIT_SECONDS)
    path = Path(output or f"artifacts/results/hybrid-kda-bench-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
