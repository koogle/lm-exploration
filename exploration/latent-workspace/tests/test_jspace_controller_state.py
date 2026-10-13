import importlib
import json
from pathlib import Path

import pytest


def fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("jspace_controller_state")


def test_restart_preserves_remote_budget_and_recovers_journaled_progress(tmp_path, monkeypatch):
    module = fixture(monkeypatch)
    handles = {
        "calibration": {"run_id": "cal01"},
        "holdout": {"call_id": "heldout"},
        "profile": {"call_id": "profile"},
    }
    protocol, calibration = {"steps": 30000}, {"fit_prompts": 1000}
    initial = {"reservations": [{"run_id": "cpu", "usd": 6}]}
    state = module.initialize(tmp_path, protocol, calibration, initial, handles)
    child = {"run_id": "train05", "phase": "train", "result": {"step": 4237, "completed": False}}
    state["children"].append(child)
    # Simulate restart between journaling completion and updating training_parent.
    (tmp_path / "state.json").write_text(json.dumps(state))
    newer = {"reservations": initial["reservations"] + [{"run_id": "train05", "usd": 5}]}
    (tmp_path / "budget.json").write_text(json.dumps(newer))
    recovered = module.initialize(tmp_path, protocol, calibration, initial, handles)
    assert recovered["training_step"] == 4237
    assert recovered["training_parent"] == "/runs/train05"
    assert json.loads((tmp_path / "budget.json").read_text()) == newer
    with pytest.raises(ValueError, match="frozen protocol"):
        module.initialize(tmp_path, {"steps": 20}, calibration, initial, handles)


def test_pending_child_survives_restart_and_completed_test_is_terminal(tmp_path, monkeypatch):
    module = fixture(monkeypatch)
    handles = {
        "calibration": {"run_id": "cal01"},
        "holdout": {"call_id": "heldout"},
        "profile": {"call_id": "profile"},
    }
    state = module.initialize(tmp_path, {}, {}, {"reservations": []}, handles)
    pending = {"phase": "train", "run_id": "train05", "call_id": "existing-call"}
    state["pending_child"] = pending
    (tmp_path / "state.json").write_text(json.dumps(state))
    recovered = module.initialize(tmp_path, {}, {}, None, handles)
    assert recovered["pending_child"] == pending
    module.apply_child(recovered, {"phase": "test", "run_id": "test01"}, {"completed": True})
    (tmp_path / "state.json").write_text(json.dumps(recovered))
    assert module.initialize(tmp_path, {}, {}, None, handles)["status"] == "completed"
