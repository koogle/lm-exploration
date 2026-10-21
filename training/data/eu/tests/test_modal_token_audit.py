from types import SimpleNamespace

import modal_token_audit as worker


def test_unknown_call_is_not_treated_as_finished(monkeypatch):
    call = SimpleNamespace(get_call_graph=list)
    monkeypatch.setattr(worker.modal.FunctionCall, "from_id", lambda _: call)
    assert worker.inspect_audit("call") == {"status": "unknown"}


def test_hub_cooldown_is_at_least_one_hour(monkeypatch):
    class RateLimit(Exception):
        response = SimpleNamespace(status_code=429, headers={"retry-after": "4200"})

    def fail(timeout):
        raise RateLimit()

    call = SimpleNamespace(
        get_call_graph=lambda: [SimpleNamespace(status=SimpleNamespace(name="FAILURE"))],
        get=fail,
    )
    monkeypatch.setattr(worker.modal.FunctionCall, "from_id", lambda _: call)
    assert worker.inspect_audit("call") == {
        "status": "failed",
        "http_status": 429,
        "retry_after_seconds": 4200,
    }


def test_terminal_graph_without_available_result_is_unresolved(monkeypatch):
    def pending(timeout):
        raise TimeoutError()

    call = SimpleNamespace(
        get_call_graph=lambda: [SimpleNamespace(status=SimpleNamespace(name="SUCCESS"))],
        get=pending,
    )
    monkeypatch.setattr(worker.modal.FunctionCall, "from_id", lambda _: call)
    assert worker.inspect_audit("call") == {"status": "unknown"}
