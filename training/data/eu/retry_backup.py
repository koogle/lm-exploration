"""Retry a verified HF code backup at the requested interval until it succeeds."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=600, help="Seconds between attempt starts")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("interval must be at least 60 seconds")
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    log = artifacts / "hf-code-backup-retries.jsonl"
    attempt = 0
    while True:
        attempt += 1
        started = datetime.now(UTC)
        deadline = time.monotonic() + args.interval
        try:
            result = subprocess.run(
                [sys.executable, "backup_code.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            success = result.returncode == 0
            detail = result.stdout if success else result.stderr[-1500:]
            transient = success or any(
                value in detail.lower()
                for value in (
                    "503",
                    "502",
                    "504",
                    "connection",
                    "timed out",
                    "timeout",
                )
            )
        except subprocess.TimeoutExpired:
            success, transient = False, True
            detail = "Attempt timed out; retry will use the same commit-addressed paths."
        entry = {
            "attempt": attempt,
            "started_at": started.isoformat(),
            "success": success,
            "next_attempt_at": None
            if success or not transient
            else (started + timedelta(seconds=args.interval)).isoformat(),
            "detail": detail,
        }
        with log.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(
            json.dumps({key: value for key, value in entry.items() if key != "detail"}), flush=True
        )
        if success:
            receipt = json.loads(result.stdout)
            if receipt.get("verified") is not True:
                raise RuntimeError("Backup did not return a verified receipt")
            (artifacts / "hf-code-backup-receipt.json").write_text(result.stdout)
            return
        if not transient:
            raise RuntimeError("Non-transient backup failure; see retry log")
        while (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(30, remaining))


if __name__ == "__main__":
    main()
