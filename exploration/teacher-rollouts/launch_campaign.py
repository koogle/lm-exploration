"""Select a measured backend, then start one bounded eight-hour campaign."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal


def read_json(volume, path):
    raw = b"".join(volume.read_file(path))
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def candidates(state):
    grouped = {}
    for trial in state.get("trials", {}).values():
        if trial.get("complete") and trial["records"] == 32:
            grouped.setdefault(trial["batch_size"], []).append(trial)
    result = []
    for size, trials in grouped.items():
        if len(trials) != 2:
            continue
        accepted = sum(t["accepted_records"] for t in trials)
        if accepted < 48:
            continue
        result.append(
            {
                "batch_size": size,
                "accepted_tokens_per_second": sum(t["accepted_tokens"] for t in trials)
                / sum(t["durable_seconds"] for t in trials),
                "accepted_records": accepted,
                "records": sum(t["records"] for t in trials),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 256:
        raise ValueError("Worker ceiling is 256")
    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    evidence = {}
    states = {}
    for backend in ["vllm", "megakernel"]:
        path = f"{args.qualification}-{backend}/producer-0/progress.json"
        states[backend], digest = read_json(volume, path)
        evidence[backend] = {"path": path, "sha256": digest}
    available = {backend: candidates(s) for backend, s in states.items()}
    if not all(available.values()):
        raise ValueError(
            "Need two complete acceptable curriculum trials from each backend"
        )
    best = {
        backend: max(rows, key=lambda r: r["accepted_tokens_per_second"])
        for backend, rows in available.items()
    }
    ratio = (
        best["megakernel"]["accepted_tokens_per_second"]
        / best["vllm"]["accepted_tokens_per_second"]
    )
    backend = "megakernel" if ratio >= 1.1 else "vllm"
    correctness_path = (
        "qwen35-targets-kernel-20261022-01/attempts/targets-kernel-01/report.json"
    )
    correctness, digest = read_json(volume, correctness_path)
    gate = correctness["buffered_distillation"]
    if not gate["correctness_passed"] or not gate["sparse_targets"]["passed"]:
        raise ValueError("Reference/target correctness gate failed")
    evidence["correctness"] = {"path": correctness_path, "sha256": digest}
    fixture_path = (
        "qwen35-targets-vllm-20261022-03/attempts/targets-vllm-03/report.json"
    )
    fixtures, digest = read_json(volume, fixture_path)
    if fixtures["status"] != "completed" or not all(
        t["accepted"] == t["requests"] for t in fixtures["trials"]
    ):
        raise ValueError("vLLM executable fixture gate failed")
    evidence["vllm_fixtures"] = {"path": fixture_path, "sha256": digest}
    config = {
        "backend": backend,
        "workers": args.workers,
        "batch_size": best[backend]["batch_size"],
        "kernel_speedup": ratio,
        "correctness_passed": True,
        "throughput_compared": True,
        "measured_candidates": best,
        "evidence": evidence,
        "launched_at": time.time(),
        "qualification_source": "7adb7d1",
        "authorization": "User requested fastest correct backend and eight hours of fresh generation",
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.launch:
        call = modal.Function.from_name(
            "lm-qwen35-distillation-teachers", "supervise"
        ).spawn(args.run_id, config)
        receipt = {"run_id": args.run_id, "call_id": call.object_id, "config": config}
        folder = Path("artifacts") / args.run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"run_id": args.run_id, "call_id": call.object_id}), flush=True
        )


if __name__ == "__main__":
    main()
