import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_restarted_controller_recovers_teacher_call_and_keeps_existing_budget(
    tmp_path, monkeypatch
):
    pytest.importorskip("modal")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("modal_long_data")
    directory = tmp_path / "controller"
    directory.mkdir()
    ledger = {"reservations": [{"run_id": "past", "gpu": "L40S", "usd": 499}]}
    (directory / "budget.json").write_text(json.dumps(ledger))
    state = {
        "phase": "extending 54k teacher trajectories",
        "runs": {},
        "package_source_sha256": "same-source",
        "teacher_run": "long-teacher-54k-r256-02",
        "teacher_call_id": "saved-call",
    }
    (directory / "state.json").write_text(json.dumps(state))
    recovered = []

    class Call:
        def get(self, timeout):
            recovered.append(timeout)
            return {
                "completed": False,
                "cursor": 35616,
                "teacher_audit": {"by_split_domain": {"train": {}}},
            }

    monkeypatch.setattr(
        module.modal.FunctionCall,
        "from_id",
        lambda ident: Call() if ident == "saved-call" else None,
    )
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if value == "/runs" else Path(value))
    monkeypatch.setattr(module, "runs", SimpleNamespace(commit=lambda: None))

    def unexpected(*args):
        raise AssertionError("recovery must not launch a duplicate teacher job")

    monkeypatch.setattr(module, "prepare", SimpleNamespace(spawn=unexpected))
    blueprint = {
        "package_source_sha256": "same-source",
        "conditions": {"fresh": {"warm_start_from": "", "continue_from": "", "steps": 100000}},
        "small_teacher": {},
    }
    result = module.grid_controller.local("controller", blueprint, {"reservations": []})
    assert recovered and result["teacher_cursor"] == 35616
    assert result["preparations"]["long-teacher-54k-r256"]["previous"] == "long-teacher-54k-r256-02"
    assert result["phase"] == "budget paused; study incomplete"
    assert json.loads((directory / "budget.json").read_text()) == ledger
