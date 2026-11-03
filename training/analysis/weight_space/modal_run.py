"""Run from training/: python analysis/weight_space/modal_run.py --help.

Every launch persists IDs and has a separate remote supervisor. Both calls have
server-side timeouts and retries disabled. The two-hour deadline includes startup.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

app = modal.App("qwen-weight-space-loop-study")
volume = modal.Volume.from_name("lm-exploration-hybrid-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.19.0", "accelerate==1.15.0")
    .env({"PYTHONPATH": "/project", "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("analysis/weight_space/study.py", "/project/weight_space/study.py")
    .add_local_file("data/qwen_agent_chat.py", "/project/training_data/qwen_agent_chat.py")
    .add_local_file("artifacts/weight-space/prompts.json", "/prompts.json")
)


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=6600, retries=0, volumes={"/runs": volume}
)
def worker(run_id, deadline, phase, selection, model_path):
    if time.time() >= deadline - 120:
        raise TimeoutError("Launch deadline reached before execution")
    from weight_space.study import run

    try:
        return run(
            model_path,
            "/prompts.json",
            f"/runs/weight-space/{run_id}",
            deadline,
            phase,
            selection,
        )
    finally:
        volume.commit()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=0.25,
    memory=512,
    timeout=7500,
    retries=0,
)
def supervisor(call_id, deadline):
    print(f"Deadline supervisor active for {call_id} until {deadline}", flush=True)
    while time.time() < deadline:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return
        except TimeoutError:
            time.sleep(min(30, max(0, deadline - time.time())))
        except Exception as error:  # noqa: BLE001 - keep the deadline guard alive on API failures
            print(f"Supervisor read failed: {type(error).__name__}", flush=True)
            time.sleep(min(30, max(0, deadline - time.time())))
    modal.FunctionCall.from_id(call_id).cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase", choices=["scan", "attention-scan", "candidates", "test"], default="scan"
    )
    parser.add_argument("--selection")
    parser.add_argument(
        "--model-path", default="/runs/agentic-reference/qwen3-0.6b-base-scale-500k-02/model"
    )
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text()) if args.selection else None
    deadline = time.time() + 7200
    out = Path("artifacts/weight-space")
    out.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    source = out / f"source-{args.phase}-{stamp}"
    source.mkdir()
    hashes = {}
    for path in Path("analysis/weight_space").glob("*.py"):
        shutil.copy2(path, source / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with modal.enable_output(), app.run(detach=True):
        call = worker.spawn(args.run_id, deadline, args.phase, selection, args.model_path)
        record = {
            "call_id": call.object_id,
            "deadline_unix": deadline,
            "phase": args.phase,
            "worker_timeout_seconds": 6600,
            "supervisor_timeout_seconds": 7500,
            "retries": 0,
            "source_sha256": hashes,
            "selection": selection,
            "model_path": args.model_path,
        }
        ledger = out / f"launch-{args.run_id}-{args.phase}-{int(time.time())}.json"
        ledger.write_text(json.dumps(record, indent=2))
        try:
            guard = supervisor.spawn(call.object_id, deadline)
            record["supervisor_id"] = guard.object_id
            ledger.write_text(json.dumps(record, indent=2))
        except BaseException:
            call.cancel()
            raise
        print(json.dumps(record), flush=True)
        result = call.get()
        (out / f"{args.phase}-result-{stamp}.json").write_text(json.dumps(result, indent=2))
        (out / f"{args.phase}-result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
