"""Verify durable dispatch claims, cooldowns, bounds and worker-state preservation."""

import copy

import pytest

from campaign import run_campaign
from hf_store import empty_state


class Store:
    def __init__(self):
        self.state = empty_state()
        self.revision = 0

    def ensure(self, private):
        assert private

    def load(self):
        return copy.deepcopy(self.state), self.revision

    def commit(self, state, revision, records, message):
        assert revision == self.revision
        assert not records
        self.state = copy.deepcopy(state)
        self.revision += 1


def test_rounds_reload_worker_state_and_wait_cooldown():
    store = Store()
    epoch = [1000]
    sleeps, calls = [], []

    def sleep(seconds):
        sleeps.append(seconds)
        epoch[0] += seconds

    def collect(run_id, max_documents, work_seconds):
        assert store.state["campaigns"]["pilot"]["rounds"][len(calls)]["status"] == "running"
        calls.append(run_id)
        store.state["completed"][run_id] = {"status": "extracted"}
        result = {"run_id": run_id, "status": "paused_rate_limit", "pending_candidates": 5}
        store.state["runs"].append(result)
        store.state["source_retry_after_epoch"] = epoch[0] + 125
        store.state["metrics"] = {"rows": len(calls)}
        store.revision += 1
        return result

    result = run_campaign("ecb", "pilot", collect, store=store, clock=lambda: epoch[0], sleep=sleep)
    assert result["status"] == "complete"
    assert len(calls) == 3
    assert len(store.state["completed"]) == 3
    assert sum(sleeps) == 250
    assert max(sleeps) <= 60
    assert result["rounds"][-1]["metrics"]["rows"] == 3
    run_campaign("ecb", "pilot", collect, store=store)
    assert len(calls) == 3


def test_crashed_dispatch_never_repeated():
    store = Store()
    calls = []

    def crash(*args):
        calls.append(args)
        raise RuntimeError("lost RPC response")

    with pytest.raises(RuntimeError):
        run_campaign("ema", "pilot", crash, store=store)
    result = run_campaign("ema", "pilot", crash, store=store)
    assert result["status"] == "ambiguous"
    assert len(calls) == 1


def test_terminal_worker_reconciles_after_lost_response():
    store = Store()

    def crash(run_id, *_):
        store.state["runs"].append(
            {"run_id": run_id, "status": "bounded_complete", "pending_candidates": 0}
        )
        store.revision += 1
        raise RuntimeError("lost response")

    with pytest.raises(RuntimeError):
        run_campaign("ema", "pilot", crash, store=store)
    result = run_campaign("ema", "pilot", crash, store=store)
    assert result["status"] == "complete"
    assert result["rounds"][0]["status"] == "complete"
    assert len(store.state["runs"]) == 1


def test_long_cooldown_stops_without_dispatch():
    store = Store()
    store.state["source_retry_after_epoch"] = 20000
    result = run_campaign("ema", "pilot", None, store=store, clock=lambda: 0)
    assert result["status"] == "deadline_reached"


def test_round_limit_validated_before_creation():
    store = Store()
    with pytest.raises(ValueError):
        run_campaign("ema", "pilot", None, rounds=4, store=store)
    assert "campaigns" not in store.state
