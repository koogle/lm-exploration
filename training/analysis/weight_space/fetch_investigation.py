"""Download committed investigation batches without waiting for other workers."""

import argparse
import json
from pathlib import Path

import modal
import truststore

truststore.inject_into_ssl()


def fetch(volume, remote, local):
    if local.exists():
        return True
    try:
        data = b"".join(volume.read_file(remote))
    except FileNotFoundError:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    config = json.loads((root / "live-config.json").read_text())
    ledger = json.loads(max(root.glob("launch-[0-9]*.json")).read_text())
    runs = modal.Volume.from_name("lm-exploration-hybrid-runs")
    data = modal.Volume.from_name("lm-exploration-hybrid-data")
    experiment = config["experiment"]
    counts = {}
    canonical = {}
    for variant, run in zip(config["variants"], ledger["runs"], strict=True):
        for seed in config["seeds"]:
            for batch in range(len(config["task_ids"]) // config["batch_tasks"]):
                canonical[variant["name"], seed, batch] = run
    remaining = list(root.glob("launch-remaining-*.json"))
    if remaining:
        assert len(remaining) == 1
        extra = json.loads(remaining[0].read_text())
        for run, schedule in zip(extra["runs"], extra["canonical_batches"], strict=True):
            for part in schedule:
                canonical["original", part["seed"], part["batch"]] = run
    for variant in config["variants"]:
        name = variant["name"]
        fetch(
            runs,
            f"weight-space/{experiment}/probes/{name}/results.json",
            root / "probes" / f"{name}.json",
        )
    committed = set()
    for run in set(canonical.values()):
        try:
            completed = json.loads(
                b"".join(runs.read_file(f"weight-space/{experiment}/{run}/completed.json"))
            )
        except FileNotFoundError:
            completed = []
        committed.update(row["run"] for row in completed)
    for (name, seed, batch_index), run in canonical.items():
        counts.setdefault(name, 0)
        label = f"seed{seed}-batch{batch_index}"
        batch = f"{run}-{label}"
        if batch not in committed:
            continue
        out = root / "live" / name / label
        remote = f"pr-tasks-pilot/agentic/{batch}"
        if fetch(data, remote + "/summary.json", out / "summary.json"):
            fetch(data, remote + "/trajectories.jsonl", out / "trajectories.jsonl")
            fetch(data, remote + "/sandboxes.json", out / "sandboxes.json")
            fetch(
                runs,
                f"weight-space/{experiment}/{run}/raw-{label}.jsonl",
                out / "raw.jsonl",
            )
            counts[name] += 1
    fetch(
        runs,
        f"weight-space/{experiment}/diagnostics/long-training-outputs.json",
        root / "long-training-outputs.json",
    )
    print(
        json.dumps(
            {
                "completed_batches": counts,
                "probes": [p.stem for p in (root / "probes").glob("*.json")],
            }
        )
    )


if __name__ == "__main__":
    main()
