"""Separate bounded workers for frozen-checkpoint context probes."""

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

app = modal.App("qwen-loop-frozen-probes")
image = (
    agentic.reference_image.add_local_file(
        "analysis/weight_space/frozen_probes.py", "/project/weight_space/frozen_probes.py"
    )
    .add_local_file("analysis/weight_space/investigate.py", "/project/weight_space/investigate.py")
    .add_local_file("analysis/weight_space/study.py", "/project/weight_space/study.py")
    .add_local_file("artifacts/weight-space/investigation/probes.json", "/probe-prompts.json")
    .add_local_file(
        "analysis/weight_space/continue_probes.py", "/project/weight_space/continue_probes.py"
    )
    .add_local_file("artifacts/weight-space/investigation/continuation.json", "/continuation.json")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=6600,
    retries=0,
    volumes={"/runs": agentic.runs_volume},
)
def worker(variant, experiment, deadline, mode):
    if mode == "continuation":
        from weight_space.continue_probes import run
    else:
        from weight_space.frozen_probes import run

    if time.time() > deadline - 120:
        raise TimeoutError("Probe launch deadline")
    try:
        return run(variant, experiment, deadline)
    finally:
        agentic.runs_volume.commit()


@app.function(image=image, cpu=0.25, memory=1024, timeout=7500, retries=0)
def supervise(calls, deadline):
    print(f"Probe supervisor active until {deadline}", flush=True)
    while time.time() < deadline:
        done = True
        for call in calls:
            try:
                modal.FunctionCall.from_id(call).get(timeout=0)
            except TimeoutError:
                done = False
            except Exception as error:  # noqa: BLE001 - retain deadline on API uncertainty
                print(type(error).__name__, flush=True)
                done = False
        if done:
            return
        time.sleep(min(30, max(0, deadline - time.time())))
    for call in calls:
        modal.FunctionCall.from_id(call).cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["probes", "continuation"], default="probes")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    stamp = int(time.time())
    deadline = time.time() + 7200
    ledger = {
        "deadline_unix": deadline,
        "timeout_seconds": 6600,
        "retries": 0,
        "prompt_sha256": hashlib.sha256(
            Path("artifacts/weight-space/investigation/probes.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("analysis/weight_space").glob("*.py")
        },
    }
    ledger["mode"] = args.mode
    if args.mode == "continuation":
        ledger["prompt_sha256"] = hashlib.sha256(
            Path("artifacts/weight-space/investigation/continuation.json").read_bytes()
        ).hexdigest()
    path = out / f"launch-{args.mode}-{stamp}.json"
    with modal.enable_output(), app.run(detach=True):
        calls = []
        try:
            for variant in config["variants"]:
                if args.mode == "continuation" and variant["name"] == "second-round":
                    continue
                calls.append(worker.spawn(variant, config["experiment"], deadline, args.mode))
                ledger["calls"] = [c.object_id for c in calls]
                path.write_text(json.dumps(ledger, indent=2))
            guard = supervise.spawn(ledger["calls"], deadline)
        except BaseException:
            for c in calls:
                c.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        (out / f"{args.mode}-result.json").write_text(
            json.dumps([c.get() for c in calls], indent=2)
        )


if __name__ == "__main__":
    main()
