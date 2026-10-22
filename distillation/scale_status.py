"""Read live sweep progress without downloading checkpoints or stopping jobs."""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime


def snapshot(run_id):
    import modal

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    volume = modal.Volume.from_name("lm-exploration-distillation")
    names = ["status.json", "plan.json"] + [
        f"producer-{i}/progress.json" for i in range(4)
    ]

    def read(name):
        value = json.loads(b"".join(volume.read_file(f"{run_id}/{name}")))
        if "chunks" in value:
            value["published_chunks"] = len(value.pop("chunks"))
        return name, value

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = dict(executor.map(read, names))
    producers = [records[f"producer-{i}/progress.json"] for i in range(4)]
    supervisor = records["status.json"]
    for name, progress in supervisor["students"].items():
        progress["execution_state"] = (
            "running"
            if name in supervisor["active_jobs"]
            else "completed"
            if progress["tokens"] >= progress["target"]
            else "waiting"
        )
        progress["token_count_kind"] = (
            "last durable checkpoint; live updates can be ahead"
        )
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "accepted_corpus_tokens": sum(p["tokens"] for p in producers),
        "accepted_records": sum(p["accepted_records"] for p in producers),
        "teacher_generated_tokens": sum(p["teacher_tokens"] for p in producers),
        "attempts": sum(p["attempts"] for p in producers),
        "rejected": sum(p["rejected"] for p in producers),
        "supervisor": supervisor,
        "plan": records["plan.json"],
        "producers": producers,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_id), indent=2))


if __name__ == "__main__":
    main()
