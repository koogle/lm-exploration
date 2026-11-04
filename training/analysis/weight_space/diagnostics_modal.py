"""Bounded read-only dataset and frozen-context probes."""

import argparse
import json
import time
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

from modal_scripts import modal_agentic as agentic

app = modal.App("qwen-loop-diagnostics")
image = agentic.reference_image.add_local_file(
    "analysis/weight_space/diagnostics.py", "/project/weight_space/diagnostics.py"
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=6600,
    retries=0,
    volumes={"/runs": agentic.runs_volume, "/hf": agentic.hf_cache},
)
def audit(run, deadline):
    from weight_space.diagnostics import audit_training

    if time.time() > deadline - 120:
        raise TimeoutError("Audit launch deadline")
    try:
        return audit_training(f"/runs/weight-space/{run}/diagnostics", deadline)
    finally:
        agentic.runs_volume.commit()


@app.function(image=image, cpu=0.25, memory=1024, timeout=7500, retries=0)
def supervise(call_id, deadline):
    print(f"Diagnostic supervisor active until {deadline}", flush=True)
    while time.time() < deadline:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return
        except TimeoutError:
            time.sleep(min(30, max(0, deadline - time.time())))
        except Exception as error:  # noqa: BLE001 - preserve deadline on uncertain status
            print(type(error).__name__, flush=True)
            time.sleep(min(30, max(0, deadline - time.time())))
    modal.FunctionCall.from_id(call_id).cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 7200
    stamp = int(time.time())
    with modal.enable_output(), app.run(detach=True):
        call = audit.spawn(args.run, deadline)
        ledger = {
            "call_id": call.object_id,
            "deadline_unix": deadline,
            "timeout_seconds": 6600,
            "retries": 0,
        }
        path = output / f"launch-audit-{stamp}.json"
        path.write_text(json.dumps(ledger, indent=2))
        try:
            guard = supervise.spawn(call.object_id, deadline)
        except BaseException:
            call.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        result = call.get()
        (output / "training-audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
