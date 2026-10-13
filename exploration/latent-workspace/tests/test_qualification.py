import importlib
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("qualification_worker")


def prompts():
    return [
        {"domain": d, "split": "test", "group": f"{d}-{i}", "prompt_ids": [j, i]}
        for j, d in enumerate(("prose", "code", "translation"))
        for i in range(64)
    ]


def test_sealed_prompt_audit_checks_source_groups_even_when_prefix_differs(worker):
    records = prompts()
    assert len(worker.audit_test_prompts(records)) == 192
    records[0]["source_group"] = "shared-article"
    records.append(
        {
            "domain": "prose",
            "split": "train",
            "group": "other-prefix",
            "source_group": "shared-article",
            "prompt_ids": [999],
        }
    )
    with pytest.raises(ValueError, match="overlaps"):
        worker.audit_test_prompts(records)


def test_sealed_prompt_audit_rejects_exact_prompt_reused_across_sources(worker):
    records = prompts()
    records.append(
        {
            "domain": "code",
            "split": "validation",
            "group": "other-repo",
            "prompt_ids": records[1]["prompt_ids"],
        }
    )
    with pytest.raises(ValueError, match="overlaps"):
        worker.audit_test_prompts(records)


def test_rollout_parity_gate_fails_closed_on_a_single_mismatch_or_empty_check(worker):
    worker.require_parity([{"identical_to_target_greedy": True}])
    for rows in ([], [{"identical_to_target_greedy": True}, {"identical_to_target_greedy": False}]):
        with pytest.raises(ValueError, match="sealed test must remain unopened"):
            worker.require_parity(rows)


def test_macro_success_cannot_hide_a_failing_domain(worker):
    m = {
        "accepted_beyond_anchor": 4,
        "by_domain": {
            d: {"mean_accepted_beyond_anchor": 2} for d in ("prose", "code", "translation")
        },
    }
    assert worker.gate(m)
    m["by_domain"]["translation"]["mean_accepted_beyond_anchor"] = 0.99
    assert not worker.gate(m)
