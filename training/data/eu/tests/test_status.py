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
