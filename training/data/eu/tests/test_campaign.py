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
        assert private is True  # Owner restored private document repositories.

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
        run_campaign("ema", "pilot", None, rounds=7, store=store)
    assert "campaigns" not in store.state


def test_spawned_call_reattached_without_duplicate_dispatch():
    from types import SimpleNamespace

    store = Store()
    spawned, loaded = [], []
    result = {"run_id": "pilot-round-1", "status": "bounded_complete", "pending_candidates": 0}

    def lost_response(**kwargs):
        entry = store.state["campaigns"]["pilot"]["rounds"][0]
        assert entry["call_id"] == "fc-existing"
        assert entry["requires_call_id"]
        raise RuntimeError("supervisor preempted")

    def spawn(*args):
        spawned.append(args)
        return SimpleNamespace(object_id="fc-existing", get=lost_response)

    def load(call_id):
        loaded.append(call_id)
        return SimpleNamespace(get=lambda **kwargs: result)

    kwargs = {"store": store, "spawn_callable": spawn, "call_loader": load}
    with pytest.raises(RuntimeError, match="preempted"):
        run_campaign("ema", "pilot", None, **kwargs)
    recovered = run_campaign("ema", "pilot", None, **kwargs)
    assert recovered["status"] == "complete"
    assert len(spawned) == 1
    assert loaded == ["fc-existing"]


def test_dispatch_gap_is_ambiguous_and_never_replayed():
    store = Store()
    calls = []

    def spawn(*args):
        calls.append(args)
        raise RuntimeError("spawn response lost before ID persisted")

    kwargs = {"store": store, "spawn_callable": spawn, "call_loader": lambda _: None}
    with pytest.raises(RuntimeError):
        run_campaign("ema", "pilot", None, **kwargs)
    assert run_campaign("ema", "pilot", None, **kwargs)["status"] == "ambiguous"
    assert len(calls) == 1


def test_gate_waits_for_id_and_legacy_passes():
    from campaign import wait_for_dispatch

    store = Store()
    wait_for_dispatch("ema", "direct", store=store)
    entry = {"run_id": "child", "status": "running", "requires_call_id": True}
    store.state["campaigns"] = {"pilot": {"rounds": [entry]}}
    epoch, sleeps = [0], []

    def sleep(seconds):
        sleeps.append(seconds)
        epoch[0] += seconds
        if epoch[0] >= 4:
            entry["call_id"] = "fc-id"

    wait_for_dispatch("ema", "child", store=store, clock=lambda: epoch[0], sleep=sleep)
    assert sleeps == [2, 2]
    del entry["call_id"]
    with pytest.raises(TimeoutError):
        wait_for_dispatch(
            "ema",
            "child",
            store=store,
            timeout_seconds=3,
            clock=lambda: epoch[0],
            sleep=lambda seconds: epoch.__setitem__(0, epoch[0] + seconds),
        )


def test_call_id_cas_retry_reloads_state_without_respawning():
    from types import SimpleNamespace

    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    store = Store()
    original_commit = store.commit
    conflicts, spawned = [], []

    def commit(state, revision, records, message):
        if message.startswith("Attach") and not conflicts:
            conflicts.append(True)
            store.state["unrelated_worker_progress"] = {"rows": 7}
            store.revision += 1
            raise HfHubHTTPError(
                "conflict",
                response=httpx.Response(409, request=httpx.Request("POST", "https://example.test")),
            )
        return original_commit(state, revision, records, message)

    store.commit = commit

    def spawn(run_id, *_):
        spawned.append(run_id)
        return SimpleNamespace(
            object_id="fc-id",
            get=lambda **kwargs: {
                "run_id": run_id,
                "status": "bounded_complete",
                "pending_candidates": 0,
            },
        )

    result = run_campaign(
        "ema",
        "pilot",
        None,
        store=store,
        spawn_callable=spawn,
        call_loader=lambda _: None,
    )
    assert result["status"] == "complete"
    assert len(spawned) == 1
    assert store.state["unrelated_worker_progress"] == {"rows": 7}


def test_predecessor_waits_without_writes():
    from campaign import wait_for_predecessor

    store = Store()
    store.state.setdefault("campaigns", {})["previous"] = {"status": "running"}
    epoch = [0]

    def sleep(seconds):
        epoch[0] += seconds
        store.state["campaigns"]["previous"]["status"] = "complete"

    wait_for_predecessor(store, "previous", clock=lambda: epoch[0], sleep=sleep)
    assert epoch[0] == 60
    assert store.revision == 0


@pytest.mark.parametrize("status", ["ambiguous", "failed", "deadline_reached"])
def test_predecessor_unsafe_states_refuse(status):
    from campaign import wait_for_predecessor

    store = Store()
    store.state.setdefault("campaigns", {})["previous"] = {"status": status}
    with pytest.raises(RuntimeError):
        wait_for_predecessor(store, "previous")
    assert store.revision == 0


def test_predecessor_wait_is_bounded():
    from campaign import wait_for_predecessor

    store = Store()
    store.state.setdefault("campaigns", {})["previous"] = {"status": "running"}
    epoch = [0]

    def sleep(seconds):
        epoch[0] += seconds

    with pytest.raises(TimeoutError):
        wait_for_predecessor(store, "previous", clock=lambda: epoch[0], sleep=sleep)
    assert epoch[0] == 1800
    assert store.revision == 0


@pytest.mark.parametrize("timeout", [0, 10801])
def test_predecessor_rejects_unbounded_wait(timeout):
    from campaign import wait_for_predecessor

    with pytest.raises(ValueError):
        wait_for_predecessor(Store(), "previous", timeout_seconds=timeout)


def test_long_predecessor_wait_allows_later_completion():
    from campaign import wait_for_predecessor

    store = Store()
    store.state["campaigns"] = {"previous": {"status": "running"}}
    epoch = [0]

    def sleep(seconds):
        epoch[0] += seconds
        if epoch[0] >= 1900:
            store.state["campaigns"]["previous"]["status"] = "complete"

    wait_for_predecessor(
        store, "previous", timeout_seconds=3600, clock=lambda: epoch[0], sleep=sleep
    )
    assert 1900 <= epoch[0] <= 1960
    assert store.revision == 0


def test_campaign_stops_when_only_unsupported_or_exhausted_items_remain():
    store = Store()
    calls = []

    def collect(run_id, max_documents, work_seconds):
        calls.append(run_id)
        return {
            "run_id": run_id,
            "status": "bounded_complete",
            "pending_candidates": 10,
            "remaining_supported_candidates": 0,
        }

    result = run_campaign("ema", "no-more-supported", collect, store=store)
    assert result["status"] == "complete"
    assert len(calls) == 1


def test_six_round_source_campaign_has_time_to_finish_and_preserves_worker_results(tmp_path):
    import json
    from pathlib import Path

    mapping = json.loads(Path("pipeline_sources.json").read_text())
    mapping["sources"]["ema"]["max_campaign_rounds"] = 6
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    store, epoch, calls = Store(), [1000], []

    def collect(run_id, maximum, seconds):
        calls.append(run_id)
        epoch[0] += 3001
        return {"run_id": run_id, "status": "bounded_complete", "pending_candidates": 100}

    result = run_campaign(
        "ema", "larger", collect, rounds=6, mapping_path=path, store=store, clock=lambda: epoch[0]
    )
    assert result["status"] == "complete"
    assert len(calls) == 6
    assert all(r["status"] == "complete" for r in result["rounds"])
    assert result["deadline_epoch"] == 1000 + 7 * 3600
