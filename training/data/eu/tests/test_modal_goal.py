from types import SimpleNamespace

import pytest

import modal_goal


@pytest.mark.parametrize(
    "saved_adapter,completed,expected",
    [("older-adapter", 4, 6), (None, 4, 6), ("current", 4, 0), ("older-adapter", 10, 0)],
)
def test_saved_zero_requires_current_adapter(monkeypatch, saved_adapter, completed, expected):
    import copy
    from pathlib import Path

    import adapters.ecb
    import hf_store
    from eu_corpus import digest

    current_hash = digest(Path(adapters.ecb.__file__).read_bytes())
    state = {
        "metrics": {"rows": completed},
        "completed": {str(n): {} for n in range(completed)},
        "collections": {"old": {"candidate_count": 10, "enumeration": {"complete": True}}},
        "runs": [
            {
                "remaining_supported_candidates": 0,
                "collection_weights": {"old": 1},
                "adapter_sha256": current_hash if saved_adapter == "current" else saved_adapter,
            }
        ],
        "campaigns": {"existing-writer": {"status": "running"}},
    }
    original = copy.deepcopy(state)
    monkeypatch.setattr(hf_store, "HFStore", lambda _: SimpleNamespace(load=lambda: (state, "rev")))
    mapping = {
        "sources": {
            "ecb": {
                "repo_id": "owner/repo",
                "adapter": "adapters.ecb",
                "collections": ["old"],
            }
        }
    }
    result = modal_goal.source_snapshots(mapping, {"config": {"sources": ["ecb"]}})["ecb"]
    assert result["remaining_supported_candidates"] == expected
    assert result["untracked_active_campaigns"] == ["existing-writer"]
    assert state == original


def node(status, children=(), call_id="call"):
    return SimpleNamespace(
        status=SimpleNamespace(name=status), children=list(children), function_call_id=call_id
    )


def test_failed_supervisor_with_live_child_is_not_safe(monkeypatch):
    call = SimpleNamespace(
        get_call_graph=lambda: [node("FAILURE", [node("PENDING", call_id="worker")])]
    )
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    assert modal_goal.inspect_call("call") == {"status": "running", "child_tree_terminated": False}


def test_completed_call_checks_result_not_only_rpc_success(monkeypatch):
    call = SimpleNamespace(
        get_call_graph=lambda: [node("SUCCESS", [node("TERMINATED", call_id="worker")])],
        get=lambda timeout: {"status": "ambiguous"},
    )
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    result = modal_goal.inspect_call("call")
    assert result["status"] == "failed"
    assert result["child_tree_terminated"] is True


def test_completed_campaign_ignores_active_sibling(monkeypatch):
    target = node("SUCCESS", [node("SUCCESS", call_id="worker")])
    ancestor = node("SUCCESS", [target, node("PENDING", call_id="sibling")], call_id="tick")
    call = SimpleNamespace(
        get_call_graph=lambda: [ancestor], get=lambda timeout: {"status": "complete"}
    )
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    assert modal_goal.inspect_call("call") == {
        "status": "complete",
        "child_tree_terminated": True,
        "result": {"status": "complete"},
    }


def test_missing_target_cannot_release_writer(monkeypatch):
    call = SimpleNamespace(get_call_graph=lambda: [node("SUCCESS", call_id="other")])
    monkeypatch.setattr(modal_goal.modal.FunctionCall, "from_id", lambda _: call)
    assert modal_goal.inspect_call("call") == {"status": "unknown", "child_tree_terminated": False}


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
        {
            "sources": {
                "ecb": {
                    "repo_id": "owner/repo",
                    "adapter": "adapters.ecb",
                    "collections": ["old", "new"],
                }
            }
        },
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
            "old": {
                "candidate_count": 5,
                "enumeration_requested_limit": 5,
                "enumeration": {"complete": False, "selection_limit": 5},
            }
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
                "adapter": "adapters.ecb",
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


def test_token_cache_read_is_pinned_and_missing_cache_counts_zero(monkeypatch):
    import json

    from huggingface_hub.errors import EntryNotFoundError

    import hf_store

    state = {
        "metrics": {"rows": 1},
        "completed": {},
        "collections": {},
        "runs": [],
        "shards": [{"path": "current", "sha256": "digest", "rows": 1}],
    }
    monkeypatch.setattr(
        hf_store, "HFStore", lambda _: SimpleNamespace(load=lambda: (state, "source-revision"))
    )
    mapping = {
        "sources": {"ecb": {"repo_id": "owner/repo", "adapter": "adapters.ecb", "collections": []}}
    }
    goal = {"config": {"sources": ["ecb"], "target_tokens": 1000}, "jobs": []}
    cache = {
        "schema_version": 1,
        "encoding": "o200k_base",
        "repo_id": "owner/repo",
        "shards": {"digest": {"path": "current", "rows": 1, "tokens": 42}},
    }

    def read(path, revision):
        assert path == "tokens/ecb.json"
        assert revision == "control-revision"
        return json.dumps(cache).encode()

    control = SimpleNamespace(_download_bytes=read)
    result = modal_goal.source_snapshots(mapping, goal, control, "control-revision")
    assert result["ecb"]["exact_tokens_lower_bound"] == 42

    def missing(path, revision):
        raise EntryNotFoundError("no audit yet")

    control._download_bytes = missing
    result = modal_goal.source_snapshots(mapping, goal, control, "control-revision")
    assert result["ecb"]["exact_tokens_lower_bound"] == 0


def test_terminal_reconciliation_loads_fresh_sources_before_second_pass(monkeypatch):
    import goal_controller

    loads, snapshots, ticks = [], [], []

    def load():
        loads.append(len(loads) + 1)
        return {"collection_goal": {"config": {"sources": ["ema"]}}}, loads[-1]

    def source_snapshots(mapping, goal, store, revision):
        snapshots.append(revision)
        return {"revision": revision}

    def run_tick(store, source, inspect, spawn, config):
        ticks.append(source["revision"])
        return {
            "jobs": [{"source_id": "ema", "status": "settling" if len(ticks) == 1 else "running"}]
        }

    monkeypatch.setattr(modal_goal, "source_snapshots", source_snapshots)
    monkeypatch.setattr(goal_controller, "run_tick", run_tick)
    result = modal_goal.reconcile_tick(
        SimpleNamespace(load=load), {}, {"sources": ["ema"]}, None, None
    )
    assert snapshots == ticks == [1, 2]
    assert result["jobs"][0]["status"] == "running"


@pytest.mark.parametrize("timestamp,expected", [(1001, True), (1599, True), (1600, False)])
def test_empty_round_backoff_expires(timestamp, expected):
    state = {
        "runs": [
            {
                "status": "bounded_complete",
                "attempted": 0,
                "remaining_supported_candidates": 1,
                "finished_at": "1970-01-01T00:16:40+00:00",
            }
        ]
    }
    assert modal_goal.retry_deferred(state, timestamp) is expected


def test_server_and_item_cooldowns_are_not_shortened():
    assert modal_goal.retry_deferred({"source_retry_after_epoch": 3000}, 2000)
    state = {
        "runs": [
            {
                "status": "bounded_complete",
                "attempted": 0,
                "remaining_supported_candidates": 1,
                "finished_at": "1970-01-01T00:16:40+00:00",
            }
        ],
        "failures": {"item": {"attempts": 2, "retry_after_epoch": 3000}},
    }
    assert modal_goal.retry_deferred(state, 2000)
    assert not modal_goal.retry_deferred(state, 3000)
