import copy
import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import EntryNotFoundError

from hf_store import RECORD_SCHEMA, HFStore, empty_state, raw_blob_path, resolve_raw_blob_path


class FakeAPI:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.files = {}
        self.sha = "initial"
        self.private = False
        self.exists = True
        self.fail = False
        self.commits = []
        self.downloads = []
        self.metadata_requests = []

    def create_repo(self, **kwargs):
        if not self.exists:
            self.private = kwargs["private"]
            self.exists = True

    def repo_info(self, **kwargs):
        if kwargs.get("files_metadata"):
            assert kwargs["revision"] == self.sha
            self.metadata_requests.append(kwargs)
        return SimpleNamespace(
            sha=self.sha,
            siblings=[
                SimpleNamespace(rfilename=path, size=len(body)) for path, body in self.files.items()
            ],
        )

    def hf_hub_download(self, filename, revision, **kwargs):
        assert revision == self.sha
        self.downloads.append(filename)
        if filename not in self.files:
            raise EntryNotFoundError("missing")
        path = self.tmp_path / "download.json"
        path.write_bytes(self.files[filename])
        return str(path)

    def create_commit(self, operations, parent_commit, **kwargs):
        if self.fail or parent_commit != self.sha:
            raise RuntimeError("commit failed or conflicted")
        pending = {op.path_in_repo: op.path_or_fileobj for op in operations}
        self.files.update(pending)
        self.commits.append(pending)
        self.sha = f"commit-{len(self.commits)}"
        return SimpleNamespace(oid=self.sha)


def record():
    return {
        "id": "doc-1",
        "source_id": "ecb",
        "collection_id": "speeches",
        "document_id": "doc-1",
        "format": "html",
        "text": "Hello world",
        "text_sha256": "abc",
        "characters": 11,
        "words": 2,
    }


def test_atomic_commit_and_resume(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    state["completed"]["doc-1"] = {"status": "complete"}
    new_revision = store.commit(state, revision, [record()])
    assert new_revision == api.sha
    assert len(api.commits) == 1
    batch = api.commits[0]
    assert {"README.md", "state/progress.json"} <= batch.keys()
    shard = state["shards"][0]["path"]
    assert shard in batch
    table = pq.read_table(io.BytesIO(batch[shard]))
    assert table.schema == RECORD_SCHEMA
    assert table["text"].to_pylist() == ["Hello world"]
    loaded, revision = store.load()
    assert loaded == state
    assert revision == new_revision
    assert b"data/speeches/**/*.parquet" in batch["README.md"]


def test_failed_commit_never_advances_remote_or_local_state(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    original = copy.deepcopy(state)
    api.fail = True
    with pytest.raises(RuntimeError):
        store.commit(state, revision, [record()])
    assert not api.files
    assert state == original


def test_conflict_not_retried(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    store.commit(state, revision, [record()])
    snapshot = dict(api.files)
    with pytest.raises(RuntimeError):
        store.commit(state, revision, [record()])
    assert api.files == snapshot
    assert len(api.commits) == 1


def test_ensure_does_not_change_existing_visibility(tmp_path):
    api = FakeAPI(tmp_path)
    HFStore("owner/ecb", api).ensure(private=True)
    assert api.private is False
    api.exists = False
    HFStore("owner/ecb", api).ensure(private=True)
    assert api.private is True


def test_duplicate_retry_is_deterministic(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    revision = store.commit(state, revision, [record()])
    path = state["shards"][0]["path"]
    data = api.files[path]
    store.commit(state, revision, [record()])
    assert len(state["shards"]) == 1
    assert api.files[path] == data
    assert not any(key.endswith(".parquet") for key in api.commits[-1])


def test_empty_batch_and_schema(tmp_path):
    assert pa.Table.from_pylist([], schema=RECORD_SCHEMA).schema == RECORD_SCHEMA
    api = FakeAPI(tmp_path)
    state = empty_state()
    HFStore("owner/ecb", api).commit(state, api.sha, [])
    assert json.loads(api.files["state/progress.json"]) == state
    assert state["shards"] == []


def test_load_auth_error_propagates(tmp_path):
    api = FakeAPI(tmp_path)

    def denied(**kwargs):
        raise PermissionError("denied")

    api.hf_hub_download = denied
    with pytest.raises(PermissionError):
        HFStore("owner/ecb", api).load()


@pytest.mark.parametrize("path", ["../secret.json", "inventories/../secret.json", "token.txt"])
def test_inventory_paths_validated(tmp_path, path):
    api = FakeAPI(tmp_path)
    with pytest.raises(ValueError):
        HFStore("owner/ecb", api).commit(empty_state(), api.sha, [], extra_files={path: b"{}"})
    assert not api.files


def test_inventory_and_raw_snapshot_in_same_commit(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    inventory_path = "inventories/speeches/" + "a" * 64 + ".json"
    raw_path = raw_blob_path("b" * 64)
    inventory = {"items": [{"url": "https://example.org/document"}]}
    revision = store.commit(
        state,
        revision,
        [record()],
        raw_files={raw_path: b"compressed-original"},
        extra_files={inventory_path: json.dumps(inventory).encode()},
    )
    assert raw_path in api.commits[0]
    assert inventory_path in api.commits[0]
    assert store.read_json(inventory_path, revision=revision) == inventory


def test_metrics_accumulate_unicode_bytes_restart_and_duplicate_retry(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    first = record() | {"text": "Grüße 世界", "raw_sha256": "c" * 64}
    raw_path = raw_blob_path("c" * 64)
    revision = store.commit(state, revision, [first], raw_files={raw_path: b"first-raw"})
    first_metrics = copy.deepcopy(state["metrics"])
    assert first_metrics["rows"] == 1
    assert first_metrics["characters"] == len("Grüße 世界")
    assert first_metrics["text_utf8_bytes"] == len("Grüße 世界".encode())
    assert first_metrics["words"] == 2
    assert first_metrics["stored_raw_gzip_bytes"] == 9
    assert first_metrics["parquet_bytes"] == len(api.files[state["shards"][0]["path"]])
    # Reload is sufficient to continue totals; no process-local accumulator exists.
    state, revision = store.load()
    second = record() | {"id": "doc-2", "collection_id": "news", "raw_sha256": "c" * 64}
    revision = store.commit(state, revision, [second], raw_files={raw_path: b"first-raw"})
    assert state["metrics"]["rows"] == 2
    assert state["metrics"]["stored_raw_gzip_bytes"] == 9
    assert state["collection_metrics"]["news"]["stored_raw_gzip_bytes"] == 0
    assert state["collection_metrics"]["speeches"]["stored_raw_gzip_bytes"] == 9
    before = copy.deepcopy(state["metrics"])
    store.commit(state, revision, [second], raw_files={raw_path: b"first-raw"})
    assert state["metrics"] == before
    assert b"Text UTF-8 bytes" in api.files["README.md"]


def test_failed_commit_does_not_advance_metrics(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    revision = store.commit(state, revision, [record()])
    before = copy.deepcopy(state)
    remote = copy.deepcopy(api.files)
    api.fail = True
    with pytest.raises(RuntimeError):
        store.commit(state, revision, [record() | {"id": "doc-2"}])
    assert state == before
    assert api.files == remote


def test_metrics_history_tracks_only_committed_growth(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    revision = store.commit(state, revision, [record()])
    assert len(state["metrics_history"]) == 1
    assert state["metrics_history"][0]["metrics"]["rows"] == 1
    assert state["metrics_history"][0]["revision_parent"] == "initial"
    revision = store.commit(state, revision, [record()])
    assert len(state["metrics_history"]) == 1
    previous_revision = revision
    revision = store.commit(state, revision, [record() | {"id": "doc-2"}])
    assert len(state["metrics_history"]) == 2
    assert state["metrics_history"][-1]["revision_parent"] == previous_revision
    assert state["metrics_history"][-1]["metrics"]["rows"] == 2
    before = copy.deepcopy(state["metrics_history"])
    api.fail = True
    with pytest.raises(RuntimeError):
        store.commit(state, revision, [record() | {"id": "doc-3"}])
    assert state["metrics_history"] == before


def http_error(status):
    import httpx

    return httpx.HTTPStatusError(
        str(status),
        request=httpx.Request("GET", "https://huggingface.co/test"),
        response=httpx.Response(status),
    )


def test_read_retries_transient_errors(tmp_path, monkeypatch):
    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    sleeps = []
    monkeypatch.setattr(hf_store.time, "sleep", sleeps.append)
    original = api.repo_info
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise http_error(503)
        return original(**kwargs)

    api.repo_info = flaky
    state, revision = store.load()
    assert revision == "initial"
    assert not state["shards"]
    assert sleeps == [1, 2]


@pytest.mark.parametrize("status", [403, 404, 409, 429])
def test_read_nontransient_status_not_retried(tmp_path, monkeypatch, status):
    import hf_store

    api = FakeAPI(tmp_path)
    calls = []
    monkeypatch.setattr(hf_store.time, "sleep", lambda delay: pytest.fail("unexpected retry"))

    def denied(**kwargs):
        calls.append(kwargs)
        raise http_error(status)

    api.repo_info = denied
    with pytest.raises(Exception, match=str(status)):
        HFStore("owner/ecb", api).load()
    assert len(calls) == 1


def test_accepted_commit_lost_response_recovers_without_duplicate(tmp_path, monkeypatch):
    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    original = api.create_commit
    monkeypatch.setattr(hf_store.time, "sleep", lambda delay: pytest.fail("unexpected retry"))

    def accepted_then_lost(**kwargs):
        original(**kwargs)
        raise http_error(503)

    api.create_commit = accepted_then_lost
    new_revision = store.commit(state, revision, [record()])
    assert new_revision == api.sha
    assert len(api.commits) == 1
    assert state["metrics"]["rows"] == 1
    assert json.loads(api.files["state/progress.json"]) == state


def test_precommit_transient_retries_same_cas_fresh_operations(tmp_path, monkeypatch):
    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    original = api.create_commit
    operations = []
    sleeps = []
    monkeypatch.setattr(hf_store.time, "sleep", sleeps.append)

    def flaky(**kwargs):
        assert kwargs["parent_commit"] == revision
        operations.append(kwargs["operations"])
        if len(operations) < 3:
            raise http_error(503)
        return original(**kwargs)

    api.create_commit = flaky
    store.commit(state, revision, [record()])
    assert sleeps == [1, 2]
    assert operations[0][0] is not operations[1][0]
    assert len(api.commits) == 1


def test_intervening_writer_not_overwritten_after_transient_error(tmp_path, monkeypatch):
    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    before = copy.deepcopy(state)
    calls = []
    monkeypatch.setattr(hf_store.time, "sleep", lambda delay: pytest.fail("unexpected retry"))

    def other_writer(**kwargs):
        calls.append(kwargs)
        api.sha = "other-writer"
        api.files["state/progress.json"] = b'{"different": true}'
        raise http_error(503)

    api.create_commit = other_writer
    with pytest.raises(Exception, match="503"):
        store.commit(state, revision, [record()])
    assert state == before
    assert len(calls) == 1
    assert api.files["state/progress.json"] == b'{"different": true}'


def test_commit_conflict_not_retried(tmp_path, monkeypatch):
    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    calls = []
    monkeypatch.setattr(hf_store.time, "sleep", lambda delay: pytest.fail("unexpected retry"))

    def conflict(**kwargs):
        calls.append(kwargs)
        raise http_error(409)

    api.create_commit = conflict
    with pytest.raises(Exception, match="409"):
        store.commit(state, revision, [record()])
    assert len(calls) == 1


def test_download_network_failure_retries_and_is_bounded(tmp_path, monkeypatch):
    import httpx

    import hf_store

    api = FakeAPI(tmp_path)
    api.files["state/progress.json"] = json.dumps(empty_state()).encode()
    store = HFStore("owner/ecb", api)
    original = api.hf_hub_download
    calls, sleeps = [], []
    monkeypatch.setattr(hf_store.time, "sleep", sleeps.append)

    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise httpx.ConnectError("disconnected")
        return original(**kwargs)

    api.hf_hub_download = flaky
    assert store.load()[1] == "initial"
    assert sleeps == [1]
    calls.clear()
    sleeps.clear()

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise httpx.ReadTimeout("unavailable")

    api.hf_hub_download = unavailable
    with pytest.raises(httpx.ReadTimeout):
        store.load()
    assert len(calls) == 3
    assert sleeps == [1, 2]


@pytest.mark.parametrize("advance_head", [False, True])
def test_busy_repository_retries_only_with_unchanged_parent(tmp_path, monkeypatch, advance_head):
    import httpx

    import hf_store

    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    before = copy.deepcopy(state)
    original = api.create_commit
    calls, sleeps = [], []
    monkeypatch.setattr(hf_store.time, "sleep", sleeps.append)

    def busy(**kwargs):
        calls.append(kwargs)
        assert kwargs["parent_commit"] == revision
        if len(calls) == 1:
            if advance_head:
                api.sha = "other-writer"
                api.files["state/progress.json"] = b'{"different": true}'
            raise httpx.HTTPStatusError(
                "409 Another commit operation is in progress for this repository.",
                request=httpx.Request("POST", "https://huggingface.co/test"),
                response=httpx.Response(409),
            )
        return original(**kwargs)

    api.create_commit = busy
    if advance_head:
        with pytest.raises(httpx.HTTPStatusError):
            store.commit(state, revision, [record()])
        assert state == before
        assert len(calls) == 1
        assert sleeps == []
    else:
        store.commit(state, revision, [record()])
        assert len(calls) == 2
        assert sleeps == [5]
        assert len(api.commits) == 1
        assert state["metrics"]["rows"] == 1


def seed_legacy_blob(store, api, raw_hash="a" * 64):
    state, revision = store.load()
    new_path = raw_blob_path(raw_hash)
    revision = store.commit(
        state, revision, [record() | {"raw_sha256": raw_hash}], raw_files={new_path: b"legacy-body"}
    )
    legacy = f"raw/{raw_hash}.gz"
    # Model a repository created by the old writer, without rewriting its shards.
    api.files[legacy] = api.files.pop(new_path)
    state["raw_blob_metrics"][legacy] = state["raw_blob_metrics"].pop(new_path)
    api.files["state/progress.json"] = json.dumps(state).encode()
    return state, revision, legacy


def test_mixed_layout_preserves_legacy_and_does_not_reupload_same_hash(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision, legacy = seed_legacy_blob(store, api)
    new_hash = "b" * 64
    revision = store.commit(
        state,
        revision,
        [
            record() | {"id": "same-raw", "collection_id": "news", "raw_sha256": "a" * 64},
            record() | {"id": "new-raw", "collection_id": "news", "raw_sha256": new_hash},
        ],
        raw_files={raw_blob_path("a" * 64): b"legacy-body", raw_blob_path(new_hash): b"new-body"},
    )
    assert resolve_raw_blob_path(state, "a" * 64) == legacy
    assert legacy in api.files
    assert raw_blob_path("a" * 64) not in api.files
    assert legacy not in api.commits[-1]
    assert raw_blob_path(new_hash) in api.commits[-1]
    assert state["metrics"]["stored_raw_gzip_bytes"] == len(b"legacy-bodynew-body")
    assert state["collection_metrics"]["news"]["stored_raw_gzip_bytes"] == len(b"new-body")
    before = copy.deepcopy(state["metrics"])
    store.commit(state, revision, [], raw_files={raw_blob_path(new_hash): b"new-body"})
    assert raw_blob_path(new_hash) not in api.commits[-1]
    assert state["metrics"] == before


def test_new_legacy_input_is_normalized_and_duplicate_aliases_are_collapsed(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    raw_hash = "a" * 64
    store.commit(
        state,
        revision,
        [record() | {"raw_sha256": raw_hash}],
        raw_files={f"raw/{raw_hash}.gz": b"body", raw_blob_path(raw_hash): b"body"},
    )
    assert f"raw/{raw_hash}.gz" not in api.files
    assert raw_blob_path(raw_hash) in api.files
    assert list(state["raw_blob_metrics"]) == [raw_blob_path(raw_hash)]
    assert state["metrics"]["stored_raw_gzip_bytes"] == 4


@pytest.mark.parametrize(
    "path", ["raw-v2/bb/" + "a" * 64 + ".gz", "raw/aa/" + "a" * 64 + ".gz", "raw-v2/../x.gz"]
)
def test_bad_partition_prefix_is_rejected_before_upload(tmp_path, path):
    api = FakeAPI(tmp_path)
    with pytest.raises(ValueError, match="Invalid raw file"):
        HFStore("owner/ecb", api).commit(empty_state(), api.sha, [], raw_files={path: b"body"})
    assert not api.commits


def test_staged_raw_is_linked_by_later_parquet_commit_without_double_count(tmp_path):
    api = FakeAPI(tmp_path)
    store = HFStore("owner/ecb", api)
    state, revision = store.load()
    raw_hash = "a" * 64
    path = raw_blob_path(raw_hash)
    state["download_queue"] = {"queued": {"raw_path": path}}
    revision = store.commit(state, revision, [], raw_files={path: b"queued-original"})
    assert state["metrics"]["rows"] == 0
    assert state["metrics"]["stored_raw_gzip_bytes"] == len(b"queued-original")
    assert state["raw_blob_metrics"][path]["shard_path"] is None
    del state["download_queue"]["queued"]
    revision = store.commit(state, revision, [record() | {"raw_sha256": raw_hash}])
    assert path not in api.commits[-1]
    assert state["metrics"]["rows"] == 1
    assert state["metrics"]["stored_raw_gzip_bytes"] == len(b"queued-original")
    assert state["raw_blob_metrics"][path]["shard_path"] == state["shards"][0]["path"]
    assert state["download_queue"] == {}
    loaded, loaded_revision = store.load()
    assert loaded_revision == revision and loaded == state
