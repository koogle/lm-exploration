"""Run not-yet-started baseline batches within the original launch deadline."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal

from analysis.weight_space.investigate_modal import app, registry, supervisor, worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    source = Path(args.ledger)
    original = json.loads(source.read_text())
    config = original["config"]
    variant = config["variants"][0]
    assert variant["name"] == "original"
    deadline = original["deadline_unix"]
    if time.time() >= deadline - 1800:
        raise TimeoutError("Insufficient original launch budget")
    # No results are selected by score. Batch 1 contains a repeatedly timing-out
    # test command, so isolate its remaining seeds to preserve cohort coverage.
    schedules = [
        [{"seed": 11, "batch": 2}]
        + [{"seed": seed, "batch": batch} for seed in (29, 47) for batch in (0, 2)],
        [{"seed": 29, "batch": 1}],
        [{"seed": 47, "batch": 1}],
    ]
    stamp = int(time.time())
    ledger = {
        "original_ledger": source.name,
        "deadline_unix": deadline,
        "timeout_seconds": 10800,
        "retries": 0,
        "canonical_batches": schedules,
        "duplicate_policy": "For these seven batches use this launch, regardless of outcomes; use the first launch for seed11 batches0 and1.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("analysis/weight_space").glob("*igate*.py")
        },
        "runs": [],
        "calls": [],
    }
    output = source.parent / f"launch-remaining-{stamp}.json"
    with modal.enable_output(), app.run(detach=True):
        calls = []
        try:
            for index, schedule in enumerate(schedules):
                run = f"{config['experiment']}-original-remaining{index}-{stamp}"
                registry[run] = []
                call = worker.spawn({**config, "batches": schedule}, variant, run, deadline)
                calls.append(call)
                ledger["runs"].append(run)
                ledger["calls"].append(call.object_id)
                output.write_text(json.dumps(ledger, indent=2))
            guard = supervisor.spawn(ledger["calls"], ledger["runs"], deadline)
        except BaseException:
            for call in calls:
                call.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        output.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        # Preserve each independent result even if another subset is incomplete.
        for index, call in enumerate(calls):
            try:
                result = call.get()
            except Exception as error:  # noqa: BLE001 - persist incomplete subsets
                result = {"error": f"{type(error).__name__}: {error}"}
            (source.parent / f"remaining-result-{index}.json").write_text(
                json.dumps(result, indent=2)
            )


if __name__ == "__main__":
    main()
