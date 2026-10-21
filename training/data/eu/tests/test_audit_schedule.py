import copy

import pytest

from audit_schedule import initial_state, run_tick


class Cache:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state)
        self.fail = False

    def load(self):
        return copy.deepcopy(self.state)

    def save(self, state):
        if self.fail:
            raise RuntimeError("CAS conflict")
        self.state = copy.deepcopy(state)


def run(cache, spawn=lambda source: source, inspect=lambda _: {"status": "complete"}, **kwargs):
    return run_tick(cache, kwargs.pop("goal_status", "collecting"), spawn, inspect, **kwargs)


def test_claims_saved_before_spawn_and_active_calls_prevent_overlap():
    cache = Cache()
    calls = []

    def spawn(source):
        assert cache.state["pairs"][0]["jobs"][source]["status"] == "claimed"
        calls.append(source)
        return f"call-{source}"

    result = run(cache, spawn)
    assert calls == ["ecb", "ema"]
    assert result["unresolved_calls"] == 2
    result = run(cache, spawn, lambda _: {"status": "running"})
    assert result["status"] == "waiting_for_calls"
    assert calls == ["ecb", "ema"]


def test_adopted_pair_consumes_budget():
    cache = Cache(initial_state({"ecb": "old-ecb", "ema": "old-ema"}))
    calls = []
    result = run(cache, lambda source: calls.append(source), max_pairs=1)
    assert result["status"] == "pair_budget_exhausted"
    assert not calls


def test_spawn_response_lost_never_replays_either_claim():
    cache = Cache()
    calls = []

    def lose(source):
        calls.append(source)
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        run(cache, lose)
    result = run(cache, lose)
    assert calls == ["ecb"]
    assert result["unresolved_calls"] == 2
    assert result["status"] == "waiting_for_calls"


def test_reservation_conflict_cannot_spawn():
    cache, calls = Cache(), []
    cache.fail = True
    with pytest.raises(RuntimeError, match="CAS"):
        run(cache, lambda source: calls.append(source))
    assert not calls


def test_failed_rate_limited_pair_waits_server_cooldown_then_resumes():
    cache = Cache(initial_state({"ecb": "ecb", "ema": "ema"}))
    inspect = lambda _: {"status": "failed", "http_status": 429, "retry_after_seconds": 7200}
    result = run(cache, inspect=inspect, clock=lambda: 100)
    assert result["status"] == "cooldown"
    assert result["cooldown_until_epoch"] == 7300
    assert run(cache, clock=lambda: 7299)["status"] == "cooldown"
    assert run(cache, clock=lambda: 7300)["pairs_reserved"] == 2


def test_rate_limit_without_header_defaults_one_hour():
    cache = Cache(initial_state({"ecb": "ecb", "ema": "ema"}))
    result = run(
        cache, inspect=lambda _: {"status": "failed", "http_status": 429}, clock=lambda: 10
    )
    assert result["cooldown_until_epoch"] == 3610


@pytest.mark.parametrize(
    "status", ["token_target_reached", "target_reached", "selected_scope_exhausted"]
)
def test_goal_stop_never_spawns(status):
    cache, calls = Cache(), []
    result = run(cache, lambda source: calls.append(source), goal_status=status)
    assert result["status"] == "goal_stopped"
    assert not calls


def test_unknown_inspection_preserves_call_and_does_not_overlap():
    cache = Cache(initial_state({"ecb": "ecb", "ema": "ema"}))
    result = run(cache, inspect=lambda _: {"status": "unknown"})
    assert result["status"] == "waiting_for_calls"
    assert cache.state["pairs"][0]["jobs"]["ecb"]["call_id"] == "ecb"
