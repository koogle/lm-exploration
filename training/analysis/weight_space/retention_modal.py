"""One bounded GPU worker compares teacher-forced retention of frozen models."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

from modal_scripts import modal_agentic as agentic

app = modal.App("qwen-loop-retention")
image = agentic.reference_image.add_local_file(
    "analysis/weight_space/retention.py", "/project/weight_space/retention.py"
).add_local_file("analysis/weight_space/investigate.py", "/project/weight_space/investigate.py")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=3300,
    retries=0,
    volumes={"/runs": agentic.runs_volume},
)
def worker(config, deadline):
    from weight_space.retention import run

    if time.time() >= deadline - 120:
        raise TimeoutError("Retention launch deadline")
    try:
        return run(config, deadline)
    finally:
        agentic.runs_volume.commit()


@app.function(image=image, cpu=0.25, memory=1024, timeout=3900, retries=0)
def supervise(call_id, deadline):
    print(f"Retention supervisor active until {deadline}", flush=True)
    while time.time() < deadline:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return
        except TimeoutError:
            pass
        except Exception as error:  # noqa: BLE001 - keep deadline on uncertain status
            print(type(error).__name__, flush=True)
        time.sleep(min(30, max(0, deadline - time.time())))
    modal.FunctionCall.from_id(call_id).cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    stamp = int(time.time())
    deadline = time.time() + 3600
    with modal.enable_output(), app.run(detach=True):
        call = worker.spawn(config, deadline)
        ledger = {
            "call_id": call.object_id,
            "deadline_unix": deadline,
            "timeout_seconds": 3300,
            "retries": 0,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("analysis/weight_space").glob("retention*.py")
            },
        }
        path = output / f"launch-retention-{stamp}.json"
        path.write_text(json.dumps(ledger, indent=2))
        try:
            guard = supervise.spawn(call.object_id, deadline)
        except BaseException:
            call.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        (output / "retention-results.json").write_text(json.dumps(call.get(), indent=2))


if __name__ == "__main__":
    main()
