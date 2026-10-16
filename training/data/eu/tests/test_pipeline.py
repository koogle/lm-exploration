"""Exercise complete collector runs against durable fake storage and fake HTTP."""

import copy
import json
from types import SimpleNamespace

import pytest

import pipeline
from hf_store import empty_state


class DurableStore:
    def __init__(self):
        self.state = empty_state()
        self.revision = 0
        self.files = {}
        self.rows = []
        self.fail_document_commit = False

    def ensure(self, private):
        assert private

    def load(self):
        return copy.deepcopy(self.state), str(self.revision)

    def read_json(self, path, revision=None):
        return json.loads(self.files[path])

    def commit(self, state, revision, records, raw_files=None, extra_files=None, message=""):
        assert revision == str(self.revision)
        if records and self.fail_document_commit:
            raise RuntimeError("simulated commit failure")
        new = copy.deepcopy(state)
        if records:
            new["shards"].append({"rows": len(records)})
        self.files.update(extra_files or {})
        self.rows.extend(copy.deepcopy(records))
        self.state = new
        self.revision += 1
        # Match HFStore's nested reference replacement after a successful commit.
        state.clear()
        state.update(copy.deepcopy(new))
        return str(self.revision)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    bodies, fetched, enumerated = {}, [], []
    inventories = {}
    mapping = {
        "private": True,
        "limits": {
            "max_documents_per_source": 1000,
            "work_seconds": 3000,
            "max_inventory_bytes": 100000,
            "max_document_bytes": 100000,
            "inventory_limit_per_collection": 1000,
            "max_attempts_per_item": 3,
            "batch_size": 2,
        },
        "sources": {
            source: {"repo_id": f"owner/{source}", "adapter": "fake", "collections": ["a", "b"]}
            for source in ("ecb", "ema")
        },
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))

    class FakeCollector:
        def __init__(self, output, max_bytes):
            output.mkdir(parents=True)
            self.client = SimpleNamespace(close=lambda: None)
            self.max_bytes = max_bytes

        def fetch(self, url, source):
            fetched.append(url)
            data = bodies[url]
            if isinstance(data, dict):
                return None, data
            return data, {
                "content_type": "text/html",
                "final_url": url,
                "retrieved_at": "2026-01-01T00:00:00Z",
                "raw_sha256": pipeline.digest(data),
            }

    def enumerate_collection(collection, fetch, limit):
        enumerated.append(collection)
        return {"records": inventories[collection], "enumeration": {"complete": True}}

    adapter = SimpleNamespace(
        __file__=pipeline.__file__,
        COLLECTIONS={"a": {}, "b": {}},
        enumerate_collection=enumerate_collection,
    )
    monkeypatch.setattr(pipeline, "Collector", FakeCollector)
    original_import = pipeline.importlib.import_module
    monkeypatch.setattr(
        pipeline.importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            adapter if name == "fake" else original_import(name, *args, **kwargs)
        ),
    )
    # Source-specific selectors would intentionally reject these synthetic HTML bodies.
    original_extract = pipeline.extract_record
    monkeypatch.setattr(
        pipeline,
        "extract_record",
        lambda candidate, data, provenance, source: original_extract(
            candidate, data, provenance, {}
        ),
    )

    def candidate(name, collection="a", source="ecb", text=None):
        url = f"https://example.org/{name}"
        bodies[url] = (
            "<html><body><article><h1>Report</h1><p>"
            + (text or (f"Institution report {name} contains detailed economic evidence. " * 20))
            + "</p></article></body></html>"
        ).encode()
        return {
            "source_id": source,
            "collection_id": collection,
            "document_id": name,
            "url": url,
            "language": "en",
            "format": "html",
            "modified_at": "2026-01-01",
        }

    def run(store, source="ecb", maximum=1000, refresh=False, run_id="test-run"):
        return pipeline.run_source(
            source,
            run_id,
            max_documents=maximum,
            work_seconds=30,
            mapping_path=mapping_path,
            artifacts_root=tmp_path / "artifacts",
            store=store,
            refresh=refresh,
        )

    return SimpleNamespace(
        bodies=bodies,
        candidate=candidate,
        run=run,
        inventories=inventories,
        fetched=fetched,
        enumerated=enumerated,
        mapping=mapping,
        adapter=adapter,
    )


def test_resume_and_round_robin_document_limit(harness):
    h = harness
    h.inventories.update(
        {
            "a": [h.candidate("a1"), h.candidate("a2")],
            "b": [h.candidate("b1", "b"), h.candidate("b2", "b")],
        }
    )
    store = DurableStore()
    first = h.run(store, maximum=2)
    assert first["attempted"] == first["extracted"] == 2, store.state["failures"]
    assert [row["document_id"] for row in store.rows] == ["a1", "b1"]
    assert len(store.state["completed"]) == 2
    second = h.run(store, maximum=2, run_id="round-two")
    assert second["attempted"] == 2
    assert [row["document_id"] for row in store.rows] == ["a1", "b1", "a2", "b2"]
    assert h.enumerated == ["a", "b"]  # Resumed immutable inventory, no network enumeration.
    assert len(h.fetched) == 4
    third = h.run(store, maximum=2, run_id="round-three")
    assert third["attempted"] == 0
    assert third["total_rows"] == 4


def test_exact_duplicates_share_persisted_record(harness):
    h = harness
    shared = "Same institutional document content and exact source wording. " * 20
    h.inventories.update(
        {
            "a": [h.candidate("first", text=shared)],
            "b": [h.candidate("mirror", "b", text=shared)],
        }
    )
    store = DurableStore()
    result = h.run(store)
    assert result["attempted"] == 2
    assert result["duplicates"] == result["extracted"] == 1
    assert len(store.rows) == 1
    entries = list(store.state["completed"].values())
    assert {entry["status"] for entry in entries} == {"extracted", "duplicate"}
    assert len({entry["record_id"] for entry in entries}) == 1
    assert h.run(store, run_id="next-round")["attempted"] == 0


def test_failed_document_commit_replays_uncommitted_documents(harness):
    h = harness
    h.inventories.update({"a": [h.candidate("a1")], "b": [h.candidate("b1", "b")]})
    store = DurableStore()
    store.fail_document_commit = True
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        h.run(store)
    assert not store.rows
    assert not store.state["completed"]
    assert not store.state["text_index"]
    assert store.state["runs"][-1]["attempted"] == 0
    assert store.state["runs"][-1]["status"] == "running"
    assert len(store.state["collections"]) == 2
    store.fail_document_commit = False
    result = h.run(store)
    assert result["extracted"] == result["total_rows"] == 2
    assert len(h.fetched) == 4
    assert h.enumerated == ["a", "b"]


def test_new_extraction_version_revisits_completed_candidates(harness, monkeypatch):
    h = harness
    h.inventories.update({"a": [h.candidate("a1")], "b": []})
    store = DurableStore()
    h.run(store)
    old_keys = set(store.state["completed"])
    monkeypatch.setattr(pipeline, "EXTRACTION_VERSION", "new-extractor")
    result = h.run(store, run_id="new-extractor-run")
    assert result["attempted"] == 1
    assert len(store.state["completed"]) == 2
    assert old_keys < store.state["completed"].keys()
    assert result["duplicates"] == 1  # Unchanged text remains one physical row.


def test_modified_source_revision_revisited_after_refresh(harness):
    h = harness
    item = h.candidate("a1")
    h.inventories.update({"a": [item], "b": []})
    store = DurableStore()
    h.run(store)
    item["modified_at"] = "2026-02-01"
    result = h.run(store, refresh=True, run_id="refresh-run")
    assert result["attempted"] == 1
    assert len(store.state["completed"]) == 2
    assert h.enumerated == ["a", "b", "a", "b"]


@pytest.mark.parametrize("source", ["ecb", "ema"])
def test_per_organization_limit_and_authorized_bound(harness, source):
    h = harness
    h.inventories.update(
        {
            "a": [h.candidate("a1", source=source), h.candidate("a2", source=source)],
            "b": [h.candidate("b1", "b", source=source)],
        }
    )
    store = DurableStore()
    result = h.run(store, source=source, maximum=1)
    assert result["attempted"] == result["total_rows"] == 1
    assert result["pending_candidates"] == 2
    with pytest.raises(ValueError, match="authorized bound"):
        h.run(store, source=source, maximum=1001)


@pytest.mark.parametrize(
    "failure",
    [
        {"http_status": 429, "error": "http_429"},
        {"error": "http_429", "headers": {"Retry-After": "1200"}},
        {"status": 429, "retry_after": "invalid"},
    ],
)
def test_rate_limit_flushes_success_and_stops_next_request(harness, monkeypatch, failure):
    h = harness
    first = h.candidate("success")
    blocked = h.candidate("limited", "b")
    untouched = h.candidate("untouched")
    h.inventories.update({"a": [first, untouched], "b": [blocked]})
    h.bodies[blocked["url"]] = failure
    monkeypatch.setattr(pipeline.time, "time", lambda: 10000)
    store = DurableStore()
    result = h.run(store)
    assert h.fetched == [first["url"], blocked["url"]]
    assert result["status"] == "paused_rate_limit"
    assert result["attempted"] == 2
    assert result["extracted"] == result["failed"] == result["total_rows"] == 1
    assert result["pending_candidates"] == 2
    expected_retry = 11200 if "headers" in failure else 10900
    assert result["retry_after_epoch"] == expected_retry
    assert store.state["source_retry_after_epoch"] == expected_retry
    assert result["source_config"]["id"] == "ecb"
    saved = store.state["failures"][pipeline.item_key(blocked)]
    assert saved["attempts"] == 1
    assert saved["retry_after_epoch"] == expected_retry
    assert pipeline.item_key(first) in store.state["completed"]
    assert pipeline.item_key(untouched) not in store.state["completed"]
    assert store.state["runs"][-1]["status"] == "paused_rate_limit"


def test_rate_limit_http_date():
    assert (
        pipeline.rate_limit_retry_epoch(
            {"headers": {"retry-after": "Thu, 01 Jan 1970 03:20:00 GMT"}}, 10000
        )
        == 12000
    )


def test_source_cooldown_stops_resume_before_network(harness, monkeypatch):
    store = DurableStore()
    store.state["source_retry_after_epoch"] = 11000
    monkeypatch.setattr(pipeline.time, "time", lambda: 10000)
    with pytest.raises(ValueError, match="rate-limited until"):
        harness.run(store)
    assert not harness.fetched
    assert not harness.enumerated
    assert not store.state["runs"]


def test_terminal_run_reinvocation_is_read_only(harness):
    h = harness
    h.inventories.update({"a": [h.candidate("a1"), h.candidate("a2")], "b": []})
    store = DurableStore()
    first = h.run(store, maximum=1)
    revision = store.revision
    repeated = h.run(store, maximum=1000, refresh=True)
    assert repeated == first
    assert store.revision == revision
    assert len(h.fetched) == 1
    assert h.enumerated == ["a", "b"]


def interrupted_run(attempted=0, started_at="1970-01-01T00:16:30+00:00"):
    return {
        "run_id": "test-run",
        "source_id": "ecb",
        "started_at": started_at,
        "status": "running",
        "max_documents": 7,
        "work_seconds": 30,
        "attempted": attempted,
        "extracted": attempted,
        "failed": 0,
        "duplicates": 0,
    }


def test_restart_reserves_uncommitted_batches_and_keeps_first_bound(harness, monkeypatch):
    h = harness
    h.inventories.update({"a": [h.candidate(f"a{i}") for i in range(10)], "b": []})
    monkeypatch.setattr(pipeline.time, "time", lambda: 1000)
    store = DurableStore()
    # Original 7-attempt authorization: 2 checkpointed + 2 possibly lost leaves 3.
    store.state["runs"].append(interrupted_run(attempted=2))
    result = h.run(store, maximum=1000)
    assert result["attempted"] == 3
    assert result["remaining_seconds_at_start"] == 20
    assert result["logical_run_attempts_persisted"] == 2
    assert result["logical_run_attempts_reserved"] == 2
    assert len(h.fetched) == 3


@pytest.mark.parametrize("reason", ["time", "attempts"])
def test_exhausted_restart_finalizes_without_source_io(harness, monkeypatch, reason):
    monkeypatch.setattr(pipeline.time, "time", lambda: 1000)
    store = DurableStore()
    old = interrupted_run(attempted=5 if reason == "attempts" else 0)
    if reason == "time":
        old["started_at"] = "1970-01-01T00:16:00+00:00"
    store.state["runs"].append(old)
    # Budget finalization takes precedence over a current source cooldown.
    store.state["source_retry_after_epoch"] = 1100
    result = harness.run(store)
    assert result["status"] == "bounded_complete"
    assert result["stop_reason"] == "logical_run_budget_exhausted"
    assert not harness.fetched
    assert not harness.enumerated
    assert len(store.state["runs"]) == 1
    assert store.revision == 1


def test_budget_sums_all_interrupted_attempts_from_first_start():
    first = interrupted_run(attempted=1)
    second = interrupted_run(attempted=1, started_at="1970-01-01T00:16:39+00:00")
    assert pipeline.run_budget([first, second], 1000, 3000, 2, 1000) == (1, 20, 2, 4)


def test_restart_uses_remaining_wall_time_for_fetch_deadline(harness, monkeypatch):
    h = harness
    h.inventories.update({"a": [h.candidate("a1"), h.candidate("a2")], "b": []})
    monkeypatch.setattr(pipeline.time, "time", lambda: 1000)
    # 10 seconds already elapsed: the new deadline is 120, not 130.
    ticks = iter([100, 100, 110, 121, 121])
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(ticks))
    store = DurableStore()
    store.state["runs"].append(interrupted_run())
    result = h.run(store)
    assert result["attempted"] == 1
    assert len(h.fetched) == 1


@pytest.mark.parametrize("status", [404, 403, 429])
def test_verified_fallback_only_after_404(harness, status):
    h = harness
    original = h.candidate("original")
    alternate = h.candidate("alternate")
    h.bodies[original["url"]] = {"http_status": status, "error": f"http_{status}"}
    h.inventories.update({"a": [original], "b": []})
    h.adapter.alternate_document_urls = lambda record: [alternate["url"]]
    store = DurableStore()
    h.run(store)
    if status == 404:
        assert h.fetched == [original["url"], alternate["url"]]
        assert pipeline.item_key(original) in store.state["completed"]
        provenance = json.loads(store.rows[0]["provenance_json"])
        assert provenance["inventory_record"]["url"] == original["url"]
        assert provenance["fetch"]["original_fetch"]["http_status"] == 404
        assert store.rows[0]["url"] == alternate["url"]
    else:
        assert h.fetched == [original["url"]]
        assert not store.rows
