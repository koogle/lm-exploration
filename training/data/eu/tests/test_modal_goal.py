from types import SimpleNamespace

import modal_goal


def node(status, children=()):
    return SimpleNamespace(status=SimpleNamespace(name=status), children=list(children))


def test_failed_supervisor_with_live_child_is_not_safe(monkeypatch):
    call = SimpleNamespace(get_call_graph=lambda: [node("FAILURE", [node("PENDING")])])
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    assert modal_goal.inspect_call("call") == {"status": "running", "child_tree_terminated": False}


def test_completed_call_checks_result_not_only_rpc_success(monkeypatch):
    call = SimpleNamespace(
        get_call_graph=lambda: [node("SUCCESS", [node("TERMINATED")])],
        get=lambda timeout: {"status": "ambiguous"},
    )
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    result = modal_goal.inspect_call("call")
    assert result["status"] == "failed"
    assert result["child_tree_terminated"] is True


def test_new_collections_are_discovery_work_not_exhaustion(monkeypatch):
    import hf_store

    state = {
        "metrics": {"rows": 10},
        "completed": {str(n): {} for n in range(10)},
        "collections": {"old": {"candidate_count": 10, "enumeration": {"complete": True}}},
        "runs": [{"remaining_supported_candidates": 0, "collection_weights": {"old": 1}}],
        "campaigns": {"manual": {"status": "running"}},
    }
    monkeypatch.setattr(
        hf_store, "HFStore", lambda _: SimpleNamespace(load=lambda: (state, "revision"))
    )
    snapshots = modal_goal.source_snapshots(
        {"sources": {"ecb": {"repo_id": "owner/repo", "collections": ["old", "new"]}}},
        {"config": {"sources": ["ecb"]}, "jobs": []},
    )
    result = snapshots["ecb"]
    assert result["discovery_pending"] is True
    assert result["inventory_complete"] is False
    assert result["untracked_active_campaigns"] == ["manual"]


def test_remote_module_import_does_not_need_build_entrypoint():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import modal, runpy, sys; modal.is_local=lambda:False; "
                "sys.modules['modal_pipeline']=None; runpy.run_path('modal_goal.py')"
            ),
        ],
        cwd=Path(modal_goal.__file__).parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_existing_capped_inventory_is_pending_expansion(monkeypatch):
    import hf_store

    state = {
        "metrics": {"rows": 5},
        "completed": {},
        "runs": [],
        "campaigns": {},
        "collections": {
            "old": {"candidate_count": 5, "enumeration": {"complete": False, "selection_limit": 5}}
        },
    }
    monkeypatch.setattr(
        hf_store, "HFStore", lambda _: SimpleNamespace(load=lambda: (state, "revision"))
    )
    mapping = {
        "limits": {"inventory_limit_per_collection": 20},
        "sources": {
            "ecb": {
                "repo_id": "owner/repo",
                "collections": ["old"],
                "inventory_limit_per_collection": 10,
            }
        },
    }
    goal = {"config": {"sources": ["ecb"]}, "jobs": []}
    assert modal_goal.source_snapshots(mapping, goal)["ecb"]["discovery_pending"] is True
    state["collections"]["old"]["enumeration_requested_limit"] = 10
    assert modal_goal.source_snapshots(mapping, goal)["ecb"]["discovery_pending"] is False


def test_recovery_identity_and_schedule_survive_remote_import():
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, EU_MODAL_APP_NAME="eu-test-recovery", EU_GOAL_SCHEDULE="0")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import modal, runpy, sys; modal.is_local=lambda:False; "
                "sys.modules['modal_pipeline']=None; m=runpy.run_path('modal_goal.py'); "
                "assert m['APP_NAME']=='eu-test-recovery'; assert m['SCHEDULE_ENABLED'] is False"
            ),
        ],
        cwd=Path(modal_goal.__file__).parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
