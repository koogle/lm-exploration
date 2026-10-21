"""Select immutable training attempts after a Modal worker restart."""

import json
from pathlib import Path


def segment_attempt(root, run, index, resume=""):
    root = Path(root)
    base = root / f"{run}-s{index:03d}"
    attempts = ([base] if base.exists() else []) + sorted(root.glob(base.name + "-retry-*"))
    candidates = []
    for folder in attempts:
        report_path = folder / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            checkpoint = Path(report["final_checkpoint"])
            if checkpoint.parent != folder or not checkpoint.is_file():
                raise ValueError("Completed attempt has no trusted final checkpoint")
            return None, str(checkpoint), report
        latest = folder / "latest.json"
        if latest.exists():
            state = json.loads(latest.read_text())
            name = state["checkpoint"]
            if Path(name).name != name or name != f"step-{state['step']:09d}.pt":
                raise ValueError("Invalid checkpoint pointer in interrupted attempt")
            checkpoint = folder / name
            if not checkpoint.is_file():
                raise ValueError("Interrupted attempt checkpoint is missing")
            candidates.append((state["step"], str(checkpoint)))
    if candidates:
        resume = max(candidates)[1]
    output = base if not attempts else root / f"{base.name}-retry-{len(attempts):03d}"
    if output.exists():
        raise ValueError("Training attempt destination already exists")
    return output, resume, None
