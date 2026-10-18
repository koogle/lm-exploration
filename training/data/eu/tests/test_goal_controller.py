"""Goal reservations and durable dispatch must survive ambiguous RPC outcomes."""

import copy

import pytest

from goal_controller import run_tick
from hf_store import empty_state


class Store:
    def __init__(self, jobs=None):
        self.state = empty_state()
        self.state["collection_goal"] = {"jobs": jobs or []}
        self.revision = 1
        self.commits = 0
        self.fail_commit = None

    def load(self):
        return copy.deepcopy(self.state), self.revision

    def commit(self, state, revision, records, message):
        self.commits += 1
        if self.commits == self.fail_commit:
            raise RuntimeError("CAS conflict")
        assert revision == self.revision and records == []
        self.state = copy.deepcopy(state)
        self.revision += 1
        return self.revision


def snapshot(rows=0, remaining=10000, **extra):
    return {"metrics": {"rows": rows}, "remaining_supported_candidates": remaining, **extra}


def tick(store, snapshots=None, inspect=None, spawn=None, config=None, now=100):
    return run_tick(
        store,
        snapshots or {"ecb": snapshot()},
        inspect or (lambda _: "running"),
        spawn or (lambda **_: "call-new"),
        config or {"sources": ["ecb"]},
        clock=lambda: now,
    )


def external(source="ecb", call="existing", status="queued"):
    return {
        "job_id": call,
        "campaign_id": call,
        "source_id": source,
        "call_id": call,
        "status": status,
        "reserved_documents": 3000,
    }


def test_failed_claim_cas_never_dispatches():
    store = Store()
    store.fail_commit = 2
    calls = []
    with pytest.raises(RuntimeError, match="CAS"):
        tick(store, spawn=lambda **kwargs: calls.append(kwargs))
    assert not calls and not store.state["collection_goal"]["jobs"]


def test_spawn_crash_gap_never_dispatches_twice():
    store, calls = Store(), []

    def spawn(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        tick(store, spawn=spawn)
    result = tick(store, spawn=spawn, now=101)
    assert len(calls) == 1
    assert result["jobs"][0]["status"] == "blocked_dispatch_gap"
    assert result["reserved_documents"] == 3000


def test_queued_existing_jobs_prevent_overlap():
    store, calls = Store([external(), external(call="next")]), []
    result = tick(store, spawn=lambda **kwargs: calls.append(kwargs), inspect=lambda _: "queued")
    assert not calls and result["reserved_documents"] == 6000


def test_target_stops_dispatch_even_with_existing_running_jobs():
    calls = []
    result = tick(
        Store([external()]),
        snapshots={"ecb": snapshot(100000)},
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls and result["status"] == "target_reached"


@pytest.mark.parametrize("extra", [{}, {"inventory_complete": True}, {"discovery_complete": True}])
def test_no_website_exhaustion_from_empty_inventory(extra):
    calls = []
    result = tick(
        Store(),
        snapshots={"ecb": snapshot(remaining=0, **extra)},
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls and result["status"] == "coverage_gap"


def test_exhaustion_requires_all_sources_and_explicit_scope_completion():
    config = {"sources": ["ecb", "ema"]}
    states = {
        key: snapshot(remaining=0, inventory_complete=True, discovery_complete=True)
        for key in config["sources"]
    }
    assert tick(Store(), snapshots=states, config=config)["status"] == "selected_scope_exhausted"
    states["ema"]["discovery_complete"] = False
    assert tick(Store(), snapshots=states, config=config)["status"] == "coverage_gap"


def test_reservations_limit_global_budget_and_partial_campaigns():
    store, calls = Store([external()]), []
    config = {"sources": ["ecb", "ema"], "target_documents": 10000}
    result = tick(
        store,
        snapshots={"ecb": snapshot(3000), "ema": snapshot(3500)},
        config=config,
        spawn=lambda **kwargs: calls.append(kwargs) or "call-ema",
    )
    assert len(calls) == 1 and calls[0]["source_id"] == "ema"
    assert calls[0]["rounds"] == 3 and calls[0]["max_documents"] == 166
    assert result["reserved_documents"] == 3498
    assert result["stored_rows"] + result["reserved_documents"] <= 10000


def test_completed_job_keeps_reservation_until_fresh_next_tick():
    store, calls = Store([external()]), []
    first = tick(store, inspect=lambda _: "complete", spawn=lambda **kwargs: calls.append(kwargs))
    assert not calls and first["jobs"][0]["status"] == "settling"
    assert first["reserved_documents"] == 3000
    second = tick(
        store,
        snapshots={"ecb": snapshot(100000)},
        now=101,
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls and second["reserved_documents"] == 0 and second["status"] == "target_reached"


def test_failed_supervisor_does_not_release_live_child_reservation():
    store, calls = Store([external()]), []
    result = tick(
        store, inspect=lambda _: {"status": "failed"}, spawn=lambda **kwargs: calls.append(kwargs)
    )
    assert not calls and result["jobs"][0]["status"] == "unknown"
    assert result["reserved_documents"] == 3000


def test_three_consecutive_confirmed_failures_block_source():
    jobs = [external(call=f"failed-{i}", status="failed") for i in range(3)]
    calls = []
    result = tick(Store(jobs), spawn=lambda **kwargs: calls.append(kwargs))
    assert not calls and result["sources"]["ecb"] == "blocked_repeated_failures"
    jobs.append(external(call="success", status="complete"))
    assert tick(Store(jobs))["jobs"][-1]["status"] == "running"


def test_untracked_source_campaign_blocks_dispatch():
    calls = []
    result = tick(
        Store(),
        snapshots={"ecb": snapshot(untracked_active_campaigns=["manual"])},
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls and result["sources"]["ecb"] == "blocked_untracked_campaign"


def test_campaign_limit_blocks_without_claiming_completion():
    calls = []
    result = tick(
        Store([external(status="complete")]),
        config={"sources": ["ecb"], "max_campaigns": 1},
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls and result["status"] == "blocked_campaign_limit"


@pytest.mark.parametrize("remaining", [None, 0])
def test_pending_discovery_dispatches_without_inventing_candidate_count(remaining):
    calls = []
    result = tick(
        Store(),
        snapshots={"ecb": snapshot(remaining=remaining, discovery_pending=True)},
        spawn=lambda **kwargs: calls.append(kwargs) or "new-discovery",
    )
    assert len(calls) == 1 and calls[0]["max_documents"] == 1000
    assert result["source_snapshots"]["ecb"]["remaining_supported_candidates"] == remaining
    assert result["reserved_documents"] == 3000


def test_lost_call_attachment_keeps_claim_and_never_respawns():
    store, calls = Store(), []
    store.fail_commit = 3  # reconciliation, claim, then call-ID attachment
    with pytest.raises(RuntimeError, match="CAS"):
        tick(store, spawn=lambda **kwargs: calls.append(kwargs) or "actually-running")
    result = tick(store, now=101, spawn=lambda **kwargs: calls.append(kwargs) or "duplicate")
    assert len(calls) == 1 and result["jobs"][0]["status"] == "blocked_dispatch_gap"


def test_failed_call_releases_only_after_confirmed_child_termination_and_refresh():
    store = Store([external()])
    result = tick(store, inspect=lambda _: {"status": "failed", "child_tree_terminated": True})
    assert result["jobs"][0]["status"] == "settling"
    result = tick(store, snapshots={"ecb": snapshot(remaining=0)}, now=101)
    assert result["jobs"][0]["status"] == "failed"
    assert result["reserved_documents"] == 0


def test_token_target_stops_new_dispatch_and_preserves_active_reservation():
    calls = []
    result = tick(
        Store([external()]),
        snapshots={
            "ecb": snapshot(100, exact_tokens_lower_bound=600),
            "ema": snapshot(100, exact_tokens_lower_bound=400),
        },
        config={"sources": ["ecb", "ema"], "target_tokens": 1000},
        spawn=lambda **kwargs: calls.append(kwargs),
    )
    assert not calls
    assert result["status"] == "token_target_reached"
    assert result["exact_tokens_lower_bound"] == 1000
    assert result["reserved_documents"] == 3000
    assert result["jobs"][0]["status"] == "running"


def test_partial_tokens_below_target_continue_dispatch():
    calls = []
    result = tick(
        Store(),
        snapshots={"ecb": snapshot(100, exact_tokens_lower_bound=999)},
        config={"sources": ["ecb"], "target_tokens": 1000},
        spawn=lambda **kwargs: calls.append(kwargs) or "new-call",
    )
    assert len(calls) == 1
    assert result["status"] == "collecting"
    assert result["exact_tokens_lower_bound"] == 999


def test_cached_tokens_are_current_shard_lower_bound():
    from goal_controller import cached_token_lower_bound

    state = {
        "shards": [
            {"path": "one", "sha256": "hash-one", "rows": 2},
            {"path": "two", "sha256": "hash-two", "rows": 1},
            {"path": "three", "sha256": "hash-three", "rows": 1},
            {"path": "four", "sha256": "hash-four", "rows": 1},
        ]
    }
    cache = {
        "schema_version": 1,
        "encoding": "o200k_base",
        "repo_id": "owner/repo",
        "shards": {
            "hash-one": {"path": "one", "rows": 2, "tokens": 25},
            "hash-two": {"path": "old-path", "rows": 1, "tokens": 500},
            "hash-three": {"path": "three", "rows": 2, "tokens": 500},
            "removed": {"path": "removed", "rows": 1, "tokens": 9999},
        },
        "summary": {"exact_tokens_counted": 99999999},
    }
    assert cached_token_lower_bound(state, cache, "owner/repo") == 25
    assert cached_token_lower_bound(state, None, "owner/repo") == 0
    with pytest.raises(ValueError, match="Incompatible"):
        cached_token_lower_bound(state, {**cache, "encoding": "other"}, "owner/repo")
    cache["shards"]["hash-one"]["tokens"] = -1
    with pytest.raises(ValueError, match="Invalid exact"):
        cached_token_lower_bound(state, cache, "owner/repo")


@pytest.mark.parametrize("target", [0, -1, True, "1000000000"])
def test_invalid_token_target_rejected(target):
    with pytest.raises(ValueError, match="target_tokens"):
        tick(Store(), config={"sources": ["ecb"], "target_tokens": target})


def test_verified_recovery_preserves_failures_and_resets_only_prior_streak():
    from goal_controller import acknowledge_recovery

    jobs = [external(call=f"failed-{i}", status="failed") for i in range(3)]
    store = Store(jobs)
    assert tick(store)["sources"]["ecb"] == "blocked_repeated_failures"
    acknowledge_recovery(store.state["collection_goal"], "ecb", "Storage layout fixed", "commit-id")
    assert all(job["status"] == "failed" for job in store.state["collection_goal"]["jobs"])
    result = tick(store)
    assert result["jobs"][-1]["call_id"] == "call-new"
    # New failures retain the original repeated-failure safeguard.
    result["jobs"][-1]["status"] = "failed"
    result["jobs"].extend([external(call=f"new-{i}", status="failed") for i in range(2)])
    store.state["collection_goal"] = result
    assert tick(store)["sources"]["ecb"] == "blocked_repeated_failures"


def test_recovery_rejects_unresolved_calls_and_missing_evidence():
    from goal_controller import acknowledge_recovery

    goal = {"config": {"sources": ["ecb"]}, "jobs": [external(status="running")]}
    with pytest.raises(ValueError, match="terminal"):
        acknowledge_recovery(goal, "ecb", "fix", "evidence")
    goal["jobs"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="evidence"):
        acknowledge_recovery(goal, "ecb", "fix", "")


def test_exhaustion_mode_keeps_collecting_after_token_milestone():
    calls = []
    result = tick(
        Store(),
        snapshots={"ecb": snapshot(100, exact_tokens_lower_bound=1001)},
        config={"sources": ["ecb"], "target_tokens": 1000, "stop_at_token_target": False},
        spawn=lambda **kwargs: calls.append(kwargs) or "new-call",
    )
    assert len(calls) == 1
    assert result["status"] == "collecting"
    assert result["exact_tokens_lower_bound"] == 1001


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_invalid_token_stop_mode_rejected(value):
    with pytest.raises(ValueError, match="stop_at_token_target"):
        tick(Store(), config={"sources": ["ecb"], "stop_at_token_target": value})


@pytest.mark.parametrize(
    "exhausted,rows,expected",
    [
        (True, 100, "selected_scope_exhausted"),
        (False, 100, "coverage_gap"),
        (False, 100000, "target_reached"),
    ],
)
def test_exhaustion_mode_preserves_coverage_requirements_and_document_budget(
    exhausted, rows, expected
):
    result = tick(
        Store(),
        snapshots={
            "ecb": snapshot(
                rows,
                remaining=0,
                inventory_complete=True,
                discovery_complete=exhausted,
                exact_tokens_lower_bound=1001,
            )
        },
        config={"sources": ["ecb"], "target_tokens": 1000, "stop_at_token_target": False},
    )
    assert result["status"] == expected
