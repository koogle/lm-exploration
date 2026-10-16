import copy
import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import EntryNotFoundError

from hf_store import RECORD_SCHEMA, HFStore, empty_state


class FakeAPI:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.files = {}
        self.sha = "initial"
        self.private = False
        self.exists = True
        self.fail = False
        self.commits = []

    def create_repo(self, **kwargs):
        if not self.exists:
            self.private = kwargs["private"]
            self.exists = True

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.sha)

    def hf_hub_download(self, filename, revision, **kwargs):
        assert revision == self.sha
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
    raw_path = "raw/" + "b" * 64 + ".gz"
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
