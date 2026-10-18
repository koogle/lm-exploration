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
