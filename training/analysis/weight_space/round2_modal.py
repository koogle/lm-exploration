"""Bounded round-two contrast, generation and export calls; run from training/."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

from modal_scripts.modal_agentic import reference_base
from workflows.images import add_project

app = modal.App("qwen-weight-space-round2")
image = (
    add_project(
        reference_base.env(
            {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                # Only our own callable and local tensor patches cross this private RPC.
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        )
    )
    .add_local_file("analysis/weight_space/study.py", "/project/weight_space/study.py")
    .add_local_file("analysis/weight_space/round2.py", "/project/weight_space/round2.py")
    .add_local_file("artifacts/weight-space/round2/prompts.json", "/round2-prompts.json")
)
volume = modal.Volume.from_name("lm-exploration-hybrid-runs")


@app.function(image=image, cpu=0.25, memory=1024, timeout=3900, retries=0)
def supervisor(call_id, deadline):
    print(f"Deadline supervisor active for {call_id} until {deadline}", flush=True)
    while time.time() < deadline:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return
        except TimeoutError:
            time.sleep(min(30, max(0, deadline - time.time())))
        except Exception as error:  # noqa: BLE001 - preserve deadline enforcement on API errors
            print(f"Supervisor read failed: {type(error).__name__}", flush=True)
            time.sleep(min(30, max(0, deadline - time.time())))
    modal.FunctionCall.from_id(call_id).cancel()


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=3300, retries=0, volumes={"/runs": volume}
)
def round2_worker(run_id, phase, selection, deadline):
    if time.time() >= deadline - 120:
        raise TimeoutError("Round-two launch deadline reached")
    from weight_space import round2

    source = "/runs/agentic-reference/qwen3-0.6b-base-scale-500k-02/model"
    out = f"/runs/weight-space/{run_id}"
    try:
        if phase == "build":
            return round2.build(source, "/round2-prompts.json", out, deadline)
        if phase == "generate":
            return round2.evaluate(source, "/round2-prompts.json", out, deadline, selection)
        return round2.export(source, out, selection)
    finally:
        volume.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["build", "generate", "export"], required=True)
    parser.add_argument("--selection")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text()) if args.selection else None
    out = Path("artifacts/weight-space/round2")
    stamp, deadline = int(time.time()), time.time() + 3600
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("analysis/weight_space").glob("*.py")
    }
    with modal.enable_output(), app.run(detach=True):
        call = round2_worker.spawn(args.run_id, args.phase, selection, deadline)
        ledger = {
            "call_id": call.object_id,
            "phase": args.phase,
            "selection": selection,
            "deadline_unix": deadline,
            "timeout_seconds": 3300,
            "retries": 0,
            "source_sha256": sources,
        }
        path = out / f"launch-{args.phase}-{stamp}.json"
        path.write_text(json.dumps(ledger, indent=2))
        try:
            guard = supervisor.spawn(call.object_id, deadline)
        except BaseException:
            call.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        result = call.get()
        (out / f"{args.phase}-result-{stamp}.json").write_text(json.dumps(result, indent=2))
        (out / f"{args.phase}-result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
