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


def test_large_curriculum_starts_two_domains_then_adds_translation_at_worker_boundary(
    tmp_path, monkeypatch
):
    pytest.importorskip("modal")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("modal_long_data")
    import time

    monkeypatch.setattr(time, "sleep", lambda _: None)
    stopped = ["fresh-grid-49152-r64", "fresh-grid-49152-r256", "fresh-grid-98304-r256"]
    names = ["fresh-grid-196608-r64", "fresh-grid-196608-r256"]
    state = {
        "package_source_sha256": "same-source",
        "runs": {n: {"status": "early stopped"} for n in stopped},
        "preparations": {
            "long-teacher-54k-r256": {"completed_run": "small-ready"},
            "long-teacher-216k": {"completed_run": "full-ready"},
        },
        "teacher_counts": {d: {"prompts": 65536} for d in ("prose", "code", "translation")},
    }
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if value == "/runs" else Path(value))
    monkeypatch.setattr(module, "runs", SimpleNamespace(commit=lambda: None))
    submitted = []

    class Call:
        def __init__(self, cfg, job):
            self.cfg, self.object_id = cfg, job

        def get(self, timeout):
            complete = bool(self.cfg.get("curriculum_switch"))
            return {
                "source_sha256": "same-source",
                "curve": [{"step": 100000 if complete else 2000}],
                "best_step": 2000,
                "training_completed": complete,
                "training_goal_step": 100000,
                "stopped": "bounded time limit; training incomplete",
                "selected_validation": {"accepted_beyond_anchor": 2},
            }

    def spawn(cfg, job):
        submitted.append((cfg, job))
        return Call(cfg, job)

    monkeypatch.setattr(module, "grid_train", SimpleNamespace(spawn=spawn))
    blueprint = {
        "package_source_sha256": "same-source",
        "resume_state": state,
        "prose_code_teacher": "prose-code-ready",
        "small_teacher": {},
        "large_teacher": {},
        "conditions": {
            n: {"warm_start_from": "", "continue_from": "", "steps": 100000}
            for n in stopped + names
        },
    }
    result = module.grid_controller.local("controller", blueprint, {"reservations": []})
    initial = [cfg for cfg, job in submitted if job.endswith("-01")]
    switched = [cfg for cfg, job in submitted if job.endswith("-02")]
    assert len(initial) == len(switched) == 2
    assert all(
        cfg["training_domains"] == ["prose", "code"] and cfg["train_limit"] == 131072
        for cfg in initial
    )
    assert all(
        cfg["training_domains"] == ["prose", "code", "translation"]
        and cfg["train_limit"] == 196608
        and cfg["steps"] == 98000
        and cfg["curriculum_switch"]
        and cfg["teacher_parent"] == "/runs/full-ready"
        for cfg in switched
    )
    assert all(result["runs"][n]["status"] == "completed" for n in names)


def test_reused_gpu_worker_reloads_volume_before_opening_new_teacher_parent(monkeypatch):
    pytest.importorskip("modal")
    module = importlib.import_module("modal_long_data")
    events = []
    monkeypatch.setattr(
        module,
        "runs",
        SimpleNamespace(
            reload=lambda: events.append("reload"), commit=lambda: events.append("commit")
        ),
    )
    monkeypatch.setattr(module, "cache", SimpleNamespace(commit=lambda: None))

    def run(cfg, output, on_checkpoint):
        assert events == ["reload"], (
            "new teacher files are invisible until the reused mount reloads"
        )
        return {"loaded_parent": cfg.teacher_parent}

    monkeypatch.setattr("latent_workspace.dflash_train.run_dflash", run)
    result = module.grid_train.local({"teacher_parent": "/runs/newly-committed-pool"}, "new-worker")
    assert result["loaded_parent"] == "/runs/newly-committed-pool"
