"""Frozen-checkpoint investigation, with a launch deadline and child cleanup."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

from modal_scripts import modal_agentic as agentic

app = modal.App("qwen-loop-investigation")
registry = modal.Dict.from_name("qwen-loop-investigation-sandboxes", create_if_missing=True)
image = agentic.reference_image.add_local_file(
    "analysis/weight_space/investigate.py", "/project/weight_space/investigate.py"
)


def cleanup(run):
    def terminate(sandbox_id):
        try:
            modal.Sandbox.from_id(sandbox_id).terminate()
        except Exception as error:  # noqa: BLE001 - idempotent cleanup of completed sandboxes
            print(f"Cleanup {sandbox_id}: {type(error).__name__}", flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(terminate, registry.get(run, [])))


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=10800,
    retries=0,
    volumes={"/runs": agentic.runs_volume, "/data": agentic.volume, "/hf": agentic.hf_cache},
)
def worker(config, variant, run, deadline):
    if time.time() >= deadline - 900:
        raise TimeoutError("Launch deadline")
    from weight_space.investigate import run_live

    try:
        return run_live(config, variant, run, deadline, app, registry)
    finally:
        cleanup(run)
        agentic.volume.commit()


@app.function(image=image, cpu=0.25, memory=1024, timeout=12000, retries=0)
def supervisor(calls, runs, deadline):
    print(f"Investigation supervisor active until {deadline}", flush=True)
    while time.time() < deadline:
        complete = True
        for call in calls:
            try:
                modal.FunctionCall.from_id(call).get(timeout=0)
            except TimeoutError:
                complete = False
            except Exception as error:  # noqa: BLE001 - preserve deadline enforcement
                print(f"Call status: {type(error).__name__}", flush=True)
                complete = False
        if complete:
            break
        time.sleep(min(30, max(0, deadline - time.time())))
    for call in calls:
        modal.FunctionCall.from_id(call).cancel()
    for run in runs:
        cleanup(run)
    return {"cleaned": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    deadline = time.time() + 11700
    runs = [f"{config['experiment']}-{v['name']}-{stamp}" for v in config["variants"]]
    for run in runs:
        registry[run] = []
    with modal.enable_output(), app.run(detach=True):
        calls = []
        ledger = {
            "config": config,
            "runs": runs,
            "deadline_unix": deadline,
            "timeout_seconds": 10800,
            "retries": 0,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("analysis/weight_space").glob("investigate*.py")
            },
        }
        ledger_path = output / f"launch-{stamp}.json"
        try:
            for variant, run in zip(config["variants"], runs, strict=True):
                calls.append(worker.spawn(config, variant, run, deadline))
                ledger["calls"] = [call.object_id for call in calls]
                ledger_path.write_text(json.dumps(ledger, indent=2))
            guard = supervisor.spawn(ledger["calls"], runs, deadline)
        except BaseException:
            for call in calls:
                call.cancel()
            for run in runs:
                cleanup(run)
            raise
        ledger["supervisor_id"] = guard.object_id
        ledger_path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        results = [call.get() for call in calls]
        (output / f"result-{stamp}.json").write_text(json.dumps(results, indent=2))
        (output / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
