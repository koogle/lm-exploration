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

    def _download_bytes(self, path, revision):
        assert revision == str(self.revision)
        return self.files[path]

    def commit(self, state, revision, records, raw_files=None, extra_files=None, message=""):
        assert revision == str(self.revision)
        if records and self.fail_document_commit:
            raise RuntimeError("simulated commit failure")
        new = copy.deepcopy(state)
        if records:
            new["shards"].append({"rows": len(records)})
        self.files.update(raw_files or {})
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
        mapping_path=mapping_path,
        collector_class=FakeCollector,
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
    ticks = iter([100, 100])
    monkeypatch.setattr(
        pipeline.time,
        "monotonic",
        lambda: next(ticks, 121 if h.fetched else 110),
    )
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


def test_weighted_schedule_preserves_coverage_and_no_repeats():
    groups = [[f"a{i}" for i in range(7)], [f"b{i}" for i in range(2)], ["c0"]]
    result = list(pipeline.round_robin(groups, [3, 1, 1]))
    assert result[:5] == ["a0", "a1", "a2", "b0", "c0"]
    assert result[5:9] == ["a3", "a4", "a5", "b1"]
    assert len(result) == len(set(result)) == 10
    assert result[-1] == "a6"


@pytest.mark.parametrize("weights", [[0], [True], [21], [1, 1]])
def test_invalid_schedule_weights(weights):
    with pytest.raises(ValueError):
        list(pipeline.round_robin([["a"]], weights))


def test_new_verified_alternate_can_recover_exhausted_original(harness):
    h = harness
    original = h.candidate("original")
    alternate = h.candidate("alternate")
    h.bodies[original["url"]] = {"http_status": 404, "error": "http_404"}
    h.inventories.update({"a": [original], "b": []})
    h.adapter.alternate_document_urls = lambda _: [alternate["url"]]
    store = DurableStore()
    key = pipeline.item_key(original)
    store.state["failures"][key] = {"attempts": 3, "error": "http_404"}
    h.run(store)
    assert key in store.state["completed"]
    assert key not in store.state["failures"]
    assert len(store.rows) == 1


def test_verified_alternate_retry_does_not_loop_or_bypass_cooldown(harness):
    h = harness
    record = h.candidate("original")
    h.adapter.alternate_document_urls = lambda _: ["https://example.org/verified.pdf"]
    store = DurableStore()
    failure = {"attempts": 3, "error": "http_404", "retry_after_epoch": 1000}
    store.state["failures"][pipeline.item_key(record)] = failure
    assert not pipeline.pending(record, store.state, timestamp=999, adapter=h.adapter)
    assert pipeline.pending(record, store.state, timestamp=1001, adapter=h.adapter)
    failure["attempted_alternates"] = ["https://example.org/verified.pdf"]
    assert not pipeline.pending(record, store.state, timestamp=1001, adapter=h.adapter)


def test_json_html_adapter_preserves_original_raw_hash(harness):
    h = harness
    candidate = h.candidate("api-document")
    html = h.bodies[candidate["url"]]
    raw = json.dumps({"html": html.decode()}).encode()
    h.bodies[candidate["url"]] = raw
    h.adapter.extract_document = lambda candidate, data: (
        json.loads(data)["html"].encode(),
        {"field": "html"},
    )
    h.inventories.update({"a": [candidate], "b": []})
    store = DurableStore()
    h.run(store)
    row = store.rows[0]
    assert row["raw_sha256"] == pipeline.digest(raw)
    assert "Institution report" in row["text"]
    provenance = json.loads(row["provenance_json"])["fetch"]
    assert provenance["adapter_transform"] == {"field": "html"}


@pytest.mark.parametrize(
    "entry,limit,expected",
    [
        (
            {
                "candidate_count": 1000,
                "enumeration_requested_limit": 1000,
                "enumeration": {"complete": False},
            },
            10000,
            True,
        ),
        (
            {
                "candidate_count": 1000,
                "enumeration_requested_limit": 10000,
                "enumeration": {"complete": False},
            },
            10000,
            False,
        ),
        (
            {
                "candidate_count": 55000,
                "enumeration_requested_limit": 100000,
                "enumeration": {"complete": False},
            },
            100000,
            False,
        ),
        (
            {
                "candidate_count": 5,
                "enumeration_requested_limit": 10,
                "enumeration": {"complete": True},
            },
            10000,
            False,
        ),
        (
            {
                "candidate_count": 5,
                "enumeration_requested_limit": 5,
                "enumeration": {"complete": False},
            },
            10,
            True,
        ),
        (
            {
                "candidate_count": 5,
                "enumeration_requested_limit": 10,
                "enumeration": {"complete": False},
            },
            10,
            False,
        ),
        (
            {
                "candidate_count": 15,
                "enumeration_requested_limit": 10,
                "enumeration": {"complete": False},
            },
            12,
            False,
        ),
    ],
)
def test_inventory_expansion_requires_larger_unattempted_limit(entry, limit, expected):
    assert pipeline.inventory_expansion_needed(entry, limit) is expected


def test_expansion_preserves_missing_candidates_provenance_and_dedup(harness):
    h = harness
    old_first, old_missing = h.candidate("first"), h.candidate("missing")
    h.inventories.update(a=[old_first, old_missing], b=[])
    store = DurableStore()
    h.run(store, maximum=1)
    previous_path = store.state["collections"]["a"]["inventory_path"]
    # Simulate an inventory selected under a lower bound.
    entry = store.state["collections"]["a"]
    entry["enumeration_requested_limit"] = 2
    entry["enumeration"] = {"complete": False, "selection_limit": 2}
    new = h.candidate("new")
    seen_limits = []

    def expanded(collection, fetch, limit):
        assert collection == "a"
        seen_limits.append(limit)
        return {
            "records": [{**old_first, "title": "Updated metadata"}, new],
            "enumeration": {"complete": False, "selection_limit": limit},
        }

    h.adapter.enumerate_collection = expanded
    result = h.run(store, run_id="expansion")
    assert result["extracted"] == 2
    assert seen_limits == [1000]
    assert len(store.rows) == 3
    assert h.fetched.count(old_first["url"]) == 1
    assert old_missing["url"] in h.fetched
    current = store.state["collections"]["a"]
    assert current["enumeration_requested_limit"] == 1000
    assert current["candidate_count"] == 3
    saved = store.read_json(current["inventory_path"])
    assert saved["expansion"]["previous_inventory_path"] == previous_path
    assert saved["expansion"]["added_candidates"] == 1
    assert previous_path in store.files
    assert saved["records"][0]["title"] == "Updated metadata"
    assert saved["enumeration"]["complete"] is False
    # Incomplete scan with fewer rows than requested is not useful to repeat.
    h.run(store, run_id="after-expansion")
    assert seen_limits == [1000]


def test_download_thread_overlaps_parsing_after_durable_stage(harness, monkeypatch):
    from threading import Event, get_ident

    h = harness
    candidates = [h.candidate(f"queued{i}") for i in range(4)]
    h.inventories.update(a=candidates, b=[])
    store = DurableStore()
    main_thread = get_ident()
    third_downloaded = Event()
    fetch_threads, commit_threads = [], []
    fetch, commit, extract = h.collector_class.fetch, store.commit, pipeline.extract_record

    def fetch_in_thread(self, url, source):
        fetch_threads.append(get_ident())
        result = fetch(self, url, source)
        if url == candidates[2]["url"]:
            third_downloaded.set()
        return result

    def commit_on_writer(*args, **kwargs):
        commit_threads.append(get_ident())
        return commit(*args, **kwargs)

    def parse_after_stage(candidate, data, provenance, source):
        assert get_ident() == main_thread
        key = pipeline.item_key(candidate)
        entry = store.state["download_queue"][key]
        assert entry["raw_path"] in store.files
        if candidate == candidates[0]:
            assert third_downloaded.wait(3), "Network producer did not overlap parsing"
        return extract(candidate, data, provenance, source)

    monkeypatch.setattr(h.collector_class, "fetch", fetch_in_thread)
    monkeypatch.setattr(store, "commit", commit_on_writer)
    monkeypatch.setattr(pipeline, "extract_record", parse_after_stage)
    result = h.run(store, maximum=4)
    assert result["attempted"] == result["extracted"] == 4
    assert result["downloaded"] == 4
    assert result["document_pipeline"] == "durable_download_queue_v1"
    assert set(fetch_threads) and main_thread not in fetch_threads
    assert set(commit_threads) == {main_thread}
    assert store.state["download_queue"] == {}


def test_parse_commit_crash_replays_raw_without_refetch(harness):
    h = harness
    h.inventories.update(a=[h.candidate("crash1"), h.candidate("crash2")], b=[])
    store = DurableStore()
    store.fail_document_commit = True
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        h.run(store, maximum=2)
    assert len(store.state["download_queue"]) == 2
    assert not store.state["completed"] and not store.rows
    assert store.state["runs"][-1]["attempted"] == 2
    original_fetches = list(h.fetched)
    h.bodies.clear()  # Network re-fetch would now fail.
    store.fail_document_commit = False
    result = h.run(store, maximum=2, run_id="parse-recovery")
    assert result["queue_replayed"] == result["extracted"] == 2
    assert h.fetched == original_fetches
    assert len(store.rows) == 2
    assert not store.state["download_queue"]


@pytest.mark.parametrize("failure", [{"http_status": 429}, {"error": "http_429"}])
def test_queued_downloader_stops_at_rate_limit(harness, failure):
    h = harness
    candidates = [h.candidate(f"limit{i}") for i in range(5)]
    h.inventories.update(a=candidates, b=[])
    h.bodies[candidates[1]["url"]] = failure
    store = DurableStore()
    result = h.run(store, maximum=5)
    assert h.fetched == [c["url"] for c in candidates[:2]]
    assert result["status"] == "paused_rate_limit"
    assert result["attempted"] == 2
    assert result["extracted"] == result["failed"] == 1
    assert store.state["source_retry_after_epoch"] > pipeline.time.time()


def test_parse_failure_retries_saved_body_with_cooldown(harness, monkeypatch):
    h = harness
    candidate = h.candidate("retryparse")
    h.inventories.update(a=[candidate], b=[])
    store = DurableStore()
    extract = pipeline.extract_record
    monkeypatch.setattr(
        pipeline, "extract_record", lambda *args: (_ for _ in ()).throw(ValueError("bad parse"))
    )
    first = h.run(store, maximum=1)
    key = pipeline.item_key(candidate)
    assert first["failed"] == 1 and key in store.state["download_queue"]
    cooldown = store.state["failures"][key]["retry_after_epoch"]
    second = h.run(store, maximum=1, run_id="during-cooldown")
    assert second["attempted"] == 0
    assert len(h.fetched) == 1
    monkeypatch.setattr(pipeline.time, "time", lambda: cooldown + 1)
    monkeypatch.setattr(pipeline, "extract_record", extract)
    third = h.run(store, maximum=1, run_id="after-cooldown")
    assert third["queue_replayed"] == third["extracted"] == 1
    assert len(h.fetched) == 1
    assert key not in store.state["failures"]


def test_queued_prefetch_keeps_authorized_attempt_bound(harness):
    h = harness
    h.inventories.update(a=[h.candidate(f"budget{i}") for i in range(7)], b=[])
    store = DurableStore()
    result = h.run(store, maximum=3)
    assert result["attempted"] == result["extracted"] == len(h.fetched) == 3
    assert result["remaining_supported_candidates"] == 4


def test_queued_alternate_validated_on_parser_thread(harness, monkeypatch):
    from threading import get_ident

    h = harness
    original, alternate = h.candidate("original"), h.candidate("alternate")
    h.inventories.update(a=[original], b=[])
    h.bodies[original["url"]] = {"http_status": 404, "error": "http_404"}
    h.adapter.alternate_document_urls = lambda _: [alternate["url"]]
    writer_thread = get_ident()
    validated = []

    def validate(candidate, body):
        assert get_ident() == writer_thread
        validated.append(candidate)

    h.adapter.validate_alternate = validate
    result = h.run(DurableStore(), maximum=1)
    assert result["extracted"] == 1
    assert validated == [original]
    assert h.fetched == [original["url"], alternate["url"]]


def test_queue_stage_failure_never_marks_download_complete(harness, monkeypatch):
    h = harness
    h.inventories.update(a=[h.candidate("stagefail")], b=[])
    store = DurableStore()
    commit = store.commit
    parsed = []

    def fail_stage(state, revision, records, raw_files=None, extra_files=None, message=""):
        if raw_files:
            raise RuntimeError("stage upload failed")
        return commit(state, revision, records, raw_files, extra_files, message)

    monkeypatch.setattr(store, "commit", fail_stage)
    monkeypatch.setattr(pipeline, "extract_record", lambda *args: parsed.append(True))
    with pytest.raises(RuntimeError, match="stage upload failed"):
        h.run(store, maximum=1)
    assert not parsed
    assert not store.state.get("download_queue")
    assert not store.state["completed"] and not store.rows


def test_queued_exact_duplicates_leave_one_parquet_record(harness):
    h = harness
    text = "Same institutional full document wording and economic evidence. " * 20
    h.inventories.update(
        a=[h.candidate("original", text=text), h.candidate("mirror", text=text)], b=[]
    )
    store = DurableStore()
    result = h.run(store, maximum=2)
    assert result["attempted"] == 2
    assert result["extracted"] == result["duplicates"] == len(store.rows) == 1
    assert not store.state["download_queue"]


def test_parse_retry_budget_does_not_loop_or_redownload(harness, monkeypatch):
    h = harness
    candidate = h.candidate("unparseable")
    h.inventories.update(a=[candidate], b=[])
    store = DurableStore()
    monkeypatch.setattr(
        pipeline, "extract_record", lambda *args: (_ for _ in ()).throw(ValueError("bad parse"))
    )
    for index in range(3):
        result = h.run(store, maximum=1, run_id=f"parse-attempt-{index}")
        assert result["failed"] == 1
        retry = store.state["failures"][pipeline.item_key(candidate)]["retry_after_epoch"]
        monkeypatch.setattr(pipeline.time, "time", lambda retry=retry: retry + 1)
    result = h.run(store, maximum=1, run_id="after-exhaustion")
    assert result["attempted"] == result["remaining_supported_candidates"] == 0
    failure = store.state["failures"][pipeline.item_key(candidate)]
    assert failure["attempts"] == 3
    # Exhausted items leave the queue; the failure keeps the staged raw reference.
    assert not store.state["download_queue"]
    assert failure["raw_path"] in store.files
    assert len(h.fetched) == 1


def test_unreadable_staged_raw_is_dropped_and_redownloaded(harness, monkeypatch):
    h = harness
    h.inventories.update(a=[h.candidate("corrupt"), h.candidate("intact")], b=[])
    store = DurableStore()
    store.fail_document_commit = True
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        h.run(store, maximum=2)
    corrupt = next(
        key
        for key, entry in store.state["download_queue"].items()
        if entry["candidate"]["document_id"] == "corrupt"
    )
    store.files[store.state["download_queue"][corrupt]["raw_path"]] = b"not gzip"
    store.fail_document_commit = False
    replayed = h.run(store, maximum=2, run_id="replay-corrupt")
    assert replayed["queue_replayed"] == 2
    assert replayed["extracted"] == replayed["failed"] == 1
    assert corrupt not in store.state["download_queue"]
    assert store.state["failures"][corrupt]["error"].startswith("staged_raw_unreadable")
    retry = store.state["failures"][corrupt]["retry_after_epoch"]
    monkeypatch.setattr(pipeline.time, "time", lambda: retry + 1)
    fetched = len(h.fetched)
    recovered = h.run(store, maximum=1, run_id="redownload-corrupt")
    assert recovered["downloaded"] == recovered["extracted"] == 1
    assert len(h.fetched) == fetched + 1


def test_parse_after_deadline_neither_commits_nor_counts():
    import document_queue

    state = {"runs": [{"attempted": 0}], "download_queue": {"k": {}}}

    class NoCommit:
        def commit(self, *args, **kwargs):
            raise AssertionError("empty checkpoint")

    revision = document_queue.parse_staged(
        NoCommit(), state, "7", ["k"], {}, None, 0, {}, replay=True
    )
    assert revision == "7"
    assert state["runs"][0]["attempted"] == 0


def test_queue_deadline_leaves_downloaded_raw_for_next_worker(harness, monkeypatch):
    import document_queue

    h = harness
    h.inventories.update(a=[h.candidate("deferredparse")], b=[])
    store = DurableStore()
    original = document_queue.parse_staged

    def defer(store, state, revision, *args, **kwargs):
        # Simulates hitting the cooperative deadline after the staging commit.
        return revision

    monkeypatch.setattr(document_queue, "parse_staged", defer)
    first = h.run(store, maximum=1)
    assert first["extracted"] == 0
    assert first["remaining_supported_candidates"] == 1
    assert len(store.state["download_queue"]) == 1
    monkeypatch.setattr(document_queue, "parse_staged", original)
    second = h.run(store, maximum=1, run_id="resume-deferred")
    assert second["queue_replayed"] == second["extracted"] == 1
    assert len(h.fetched) == 1


def test_interrupted_queue_reserves_one_speculative_batch(harness):
    h = harness
    h.inventories.update(a=[h.candidate(f"preempt{i}") for i in range(6)], b=[])
    store = DurableStore()
    store.fail_document_commit = True
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        h.run(store, maximum=6)
    # First batch is durable; no more than one additional batch could be fetched.
    assert store.state["runs"][-1]["attempted"] == 2
    assert 2 <= len(h.fetched) <= 4
    fetches = list(h.fetched)
    store.fail_document_commit = False
    resumed = h.run(store, maximum=6)
    assert resumed["remaining_documents_at_start"] == 2
    assert resumed["logical_run_attempts_persisted"] == 2
    assert resumed["logical_run_attempts_reserved"] == 2
    assert resumed["queue_replayed"] == resumed["extracted"] == 2
    assert h.fetched == fetches


def test_mapping_collections_default_to_adapter_registry():
    import importlib

    mapping = pipeline.load_mapping()
    ecb, ema = mapping["sources"]["ecb"], mapping["sources"]["ema"]
    assert ecb["collections"] == list(importlib.import_module("adapters.ecb").COLLECTIONS)
    assert "epar_documents" not in ema["collections"]
    assert len(ema["collections"]) == len(importlib.import_module("adapters.ema").COLLECTIONS) - 1
    for config in mapping["sources"].values():
        assert set(config.get("collection_weights", {})) <= set(config["collections"])
