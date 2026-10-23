"""Preflight and reserve before initializing Modal (which may build images)."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from core import atomic_json, reserve_budget, validate_config

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument(
        "--backend",
        choices=["baseline", "megakernel", "kernel-sweep", "kernel-pipeline"],
        default="baseline",
    )
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    pinned = json.loads((HERE / "config.json").read_text())
    validate_config(c)
    for key in ("image", "runtime", "model", "revision", "dtype"):
        if c[key] != pinned[key]:
            raise ValueError(f"{key} differs from pinned build config")
    for name in (args.run_id, args.attempt):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("invalid ID")
    manifest = {
        "backend": args.backend,
        "config": c,
        "prompts": json.loads((HERE / "prompts.json").read_text()),
        "tokenizer_revision": c["revision"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                p
                for p in HERE.rglob("*")
                if p.is_file()
                and p.suffix in (".py", ".cu", ".h", ".lock")
                and not any(
                    part.startswith(".") or part == "artifacts"
                    for part in p.relative_to(HERE).parts
                )
            )
            + [HERE / "prompts.json", HERE / "requirements.lock"]
        },
    }
    if not args.launch:
        print(json.dumps({"mode": "dry_run", "manifest": manifest}, indent=2))
        return
    reserve_budget(c, HERE / ".budget", args.attempt)
    # Import and app startup only after authorization and budget reservation.
    if os.environ.get("CODEX_PROXY_CERT"):
        import truststore

        truststore.inject_into_ssl()
    import modal

    os.environ["ROLLOUT_BACKEND"] = args.backend
    from modal_app import app, supervision, supervisor

    started = time.monotonic()
    deadline = time.time() + c["runtime_seconds"] + 60
    out = HERE / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"{args.attempt}-manifest.json", manifest)
    atomic_json(out / f"{args.attempt}-launch.json", {"deadline_unix": deadline})
    report = None
    with modal.enable_output(), app.run():
        call = supervisor.spawn(args.run_id, args.attempt, manifest, deadline)
        atomic_json(
            out / f"{args.attempt}-launch.json",
            {"deadline_unix": deadline, "supervisor_call_id": call.object_id},
        )
        try:
            report = call.get()
        finally:
            state = supervision.get(f"{args.run_id}/{args.attempt}")
            if state:
                atomic_json(out / f"{args.attempt}-supervision.json", state)
    if report is None:
        raise RuntimeError(
            "Run interrupted before a report was returned; inspect persisted supervision state"
        )
    report["client_total_seconds_including_build_and_queue"] = (
        time.monotonic() - started
    )
    atomic_json(out / f"{args.attempt}-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
