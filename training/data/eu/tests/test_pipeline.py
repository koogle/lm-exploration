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
    monkeypatch.setattr(pipeline.importlib, "import_module", lambda name: adapter)
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

    def run(store, source="ecb", maximum=1000, refresh=False):
        return pipeline.run_source(
            source,
            "test-run",
            max_documents=maximum,
            work_seconds=30,
            mapping_path=mapping_path,
            artifacts_root=tmp_path / "artifacts",
            store=store,
            refresh=refresh,
        )

    return SimpleNamespace(
        candidate=candidate,
        run=run,
        inventories=inventories,
        fetched=fetched,
        enumerated=enumerated,
        mapping=mapping,
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
    assert first["attempted"] == first["extracted"] == 2
    assert [row["document_id"] for row in store.rows] == ["a1", "b1"]
    assert len(store.state["completed"]) == 2
    second = h.run(store, maximum=2)
    assert second["attempted"] == 2
    assert [row["document_id"] for row in store.rows] == ["a1", "b1", "a2", "b2"]
    assert h.enumerated == ["a", "b"]  # Resumed immutable inventory, no network enumeration.
    assert len(h.fetched) == 4
    third = h.run(store, maximum=2)
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
    assert h.run(store)["attempted"] == 0


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
    result = h.run(store)
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
    result = h.run(store, refresh=True)
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
