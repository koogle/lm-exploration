"""Idempotent controller recovery; preserve reservations and completed child progress."""

import json
from pathlib import Path


def apply_child(state, child, result):
    if child["phase"] == "calibration":
        state["calibration_result"] = result
        state["calibration_parent"] = f"/runs/{child['run_id']}"
        state["calibrated_prompts"] = result["counts"]
    elif child["phase"] == "train":
        state["training_result"] = result
        state["training_parent"] = f"/runs/{child['run_id']}"
        state["training_step"] = result["step"]
    elif child["phase"] == "test":
        state["status"] = "completed"
        state["result"] = result
    else:
        raise ValueError("Unknown recovered child phase")


def initialize(root, protocol, calibration_protocol, ledger, handles, continuation=None):
    root = Path(root)
    root.mkdir(exist_ok=True)
    budget_path = root / "budget.json"
    if not budget_path.exists():
        if ledger is None:
            raise ValueError("Missing authoritative ledger on controller recovery")
        budget_path.write_text(json.dumps(ledger, indent=2) + "\n")
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    elif continuation is not None:
        state = continuation
    else:
        state = {
            "status": "running",
            "phase": "calibration",
            "children": [],
            "protocol": protocol,
            "calibration_protocol": calibration_protocol,
            "calibration_parent": f"/runs/{handles['calibration']['run_id']}",
            "training_parent": "",
            "calibration_chunks": 1,
            "training_chunks": 0,
            "heldout_call_id": handles["holdout"]["call_id"],
            "profile_call_id": handles["profile"]["call_id"],
        }
    if state["protocol"] != protocol or state["calibration_protocol"] != calibration_protocol:
        raise ValueError("Controller recovery changes the frozen protocol")
    # A restart may occur after child completion was journaled but before parent fields updated.
    for child in state.get("children", []):
        apply_child(state, child, child["result"])
    if state["status"] != "completed":
        state["status"] = "running"
        state.pop("error", None)
    return state
