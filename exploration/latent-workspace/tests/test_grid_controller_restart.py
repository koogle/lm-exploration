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


def test_restarted_controller_does_not_relaunch_early_stopped_small_fits(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("modal_long_data")
    names = ["fresh-grid-49152-r64", "fresh-grid-49152-r256"]
    state = {
        "package_source_sha256": "same-source",
        "runs": {name: {"status": "early stopped"} for name in names},
        "preparations": {"long-teacher-54k-r256": {"completed_run": "small-ready"}},
    }
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if value == "/runs" else Path(value))
    monkeypatch.setattr(module, "runs", SimpleNamespace(commit=lambda: None))

    def unexpected(*args):
        raise AssertionError("early-stopped drafts must not restart")

    monkeypatch.setattr(module, "grid_train", SimpleNamespace(spawn=unexpected))
    monkeypatch.setattr(module, "prepare", SimpleNamespace(spawn=unexpected))
    blueprint = {
        "package_source_sha256": "same-source",
        "resume_state": state,
        "conditions": {
            name: {"warm_start_from": "", "continue_from": "", "steps": 100000} for name in names
        },
        "small_teacher": {},
        "large_teacher": {},
    }
    ledger = {"reservations": [{"run_id": "prior", "gpu": "L40S", "usd": 499}]}
    result = module.grid_controller.local("controller", blueprint, ledger)
    assert all(result["runs"][name]["status"] == "early stopped" for name in names)
    assert result["phase"] == "budget paused; study incomplete"


def test_controller_starts_available_98k_subset_before_full_preparation(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("modal_long_data")
    small_names = ["fresh-grid-49152-r64", "fresh-grid-49152-r256"]
    name = "fresh-grid-98304-r256"
    state = {
        "package_source_sha256": "same-source",
        "runs": {n: {"status": "early stopped"} for n in small_names},
        "preparations": {"long-teacher-54k-r256": {"completed_run": "small-ready"}},
    }
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if value == "/runs" else Path(value))
    monkeypatch.setattr(module, "runs", SimpleNamespace(commit=lambda: None))
    submitted = []

    class Call:
        object_id = "draft-call"

        def get(self, timeout):
            return {
                "source_sha256": "same-source",
                "curve": [{"step": 100000}],
                "best_step": 100000,
                "training_completed": True,
                "selected_validation": {"accepted_beyond_anchor": 2},
            }

    def spawn(cfg, run_id):
        submitted.append((cfg, run_id))
        return Call()

    def reserve(path, run_id, *args, **kwargs):
        if run_id.startswith("long-teacher-216k"):
            raise ValueError("allocation cap reached")

    monkeypatch.setattr(module, "reserve", reserve)
    monkeypatch.setattr(module, "grid_train", SimpleNamespace(spawn=spawn))
    blueprint = {
        "package_source_sha256": "same-source",
        "resume_state": state,
        "available_teacher": "sealed-98k",
        "conditions": {
            n: {"warm_start_from": "", "continue_from": "", "steps": 100000}
            for n in [*small_names, name]
        },
        "small_teacher": {},
        "large_teacher": {},
    }
    result = module.grid_controller.local("controller", blueprint, {"reservations": []})
    assert len(submitted) == 1
    assert submitted[0][0]["teacher_parent"] == "/runs/sealed-98k"
    assert submitted[0][0]["warm_start_from"] == ""
    assert result["runs"][name]["status"] == "completed"
