import json

import pytest
from huggingface_hub.errors import EntryNotFoundError

from status import cached_token_status


class Control:
    def __init__(self, cache):
        self.cache = cache

    def _download_bytes(self, path, revision):
        assert path == "tokens/ecb.json"
        assert revision == "pinned-control"
        if self.cache is None:
            raise EntryNotFoundError("missing")
        return json.dumps(self.cache).encode()


def test_cached_status_excludes_stale_entries_despite_complete_old_summary():
    state = {
        "shards": [
            {"path": "a", "sha256": "hash-a", "rows": 2},
            {"path": "b", "sha256": "hash-b", "rows": 3},
            {"path": "c", "sha256": "hash-c", "rows": 4},
        ]
    }
    cache = {
        "schema_version": 1,
        "encoding": "o200k_base",
        "repo_id": "owner/ecb",
        "summary": {"complete_for_revision": True, "exact_tokens_counted": 999},
        "shards": {
            "hash-a": {"path": "a", "rows": 2, "tokens": 17},
            "hash-b": {"path": "old-b", "rows": 3, "tokens": 800},
            "hash-c": {"path": "c", "rows": 9, "tokens": 182},
        },
    }
    result = cached_token_status(state, "owner/ecb", "ecb", Control(cache), "pinned-control")
    assert result["exact_tokens_lower_bound"] == 17
    assert result["matched_rows"] == 2
    assert not result["complete_for_source_revision"]
    assert result["total_shards"] == 3


def test_missing_cache_is_explicit_not_complete():
    result = cached_token_status(
        {"shards": []}, "owner/ecb", "ecb", Control(None), "pinned-control"
    )
    assert result["exact_tokens_lower_bound"] == 0
    assert not result["cache_present"]
    assert not result["complete_for_source_revision"]


def test_invalid_cache_is_not_silently_counted():
    with pytest.raises(ValueError, match="Incompatible"):
        cached_token_status({"shards": []}, "owner/ecb", "ecb", Control({}), "pinned-control")


def document_target_fixture():
    target = {
        "schema_version": 1,
        "request_id": "another-10000",
        "baseline_total_rows": 20703,
        "additional_documents": 10000,
        "target_total_rows": 30703,
        "sources": [
            {"source_id": "ecb", "repo_id": "owner/ecb", "baseline_rows": 11671},
            {"source_id": "ema", "repo_id": "owner/ema", "baseline_rows": 9032},
        ],
    }
    sources = [
        {"source_id": "ecb", "repo_id": "owner/ecb", "rows": 11700, "revision": "ecb-pinned"},
        {"source_id": "ema", "repo_id": "owner/ema", "rows": 9200, "revision": "ema-pinned"},
    ]
    return target, sources


def test_document_target_excludes_baseline_and_keeps_revision_evidence():
    from status import document_target_status

    target, sources = document_target_fixture()
    result = document_target_status(target, sources)
    assert result["added_rows"] == 197
    assert result["remaining_rows"] == 9803
    assert result["status"] == "collecting"
    assert result["source_revisions"] == {"ecb": "ecb-pinned", "ema": "ema-pinned"}


@pytest.mark.parametrize("delta,reached", [(-1, False), (0, True), (1, True)])
def test_document_target_requires_all_additional_rows(delta, reached):
    from status import document_target_status

    target, sources = document_target_fixture()
    sources[1]["rows"] = 30703 - sources[0]["rows"] + delta
    result = document_target_status(target, sources)
    assert (result["status"] == "reached") is reached
    assert result["remaining_rows"] == max(0, -delta)


@pytest.mark.parametrize("failure", ["missing", "read_error", "wrong_repo", "regression"])
def test_document_target_cannot_complete_from_incompatible_source_reads(failure):
    from status import document_target_status

    target, sources = document_target_fixture()
    sources[0]["rows"] = 50000
    if failure == "missing":
        sources.pop()
    elif failure == "read_error":
        sources[1] = {"source_id": "ema", "error_type": "TimeoutError"}
    elif failure == "wrong_repo":
        sources[1]["repo_id"] = "different/repo"
    else:
        sources[1]["rows"] = 9031
    with pytest.raises(ValueError):
        document_target_status(target, sources)


def test_document_target_rejects_changed_baseline_totals():
    from status import document_target_status

    target, sources = document_target_fixture()
    target["baseline_total_rows"] -= 1
    target["target_total_rows"] -= 1
    with pytest.raises(ValueError, match="baseline total"):
        document_target_status(target, sources)
