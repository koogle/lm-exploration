import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import EntryNotFoundError

from hf_store import RECORD_SCHEMA, json_bytes
from token_audit import AuditCache, measure_shard, run_audit


class Encoding:
    name = "o200k_base"

    def encode(self, text, *, disallowed_special):
        assert disallowed_special == ()
        return list(text)  # Deliberately transparent fake, not a real token forecast.


def make_shard(text, number):
    row = dict.fromkeys(RECORD_SCHEMA.names)
    row.update(text=text, id=str(number))
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist([row], schema=RECORD_SCHEMA), buffer)
    digest = hashlib.sha256(json_bytes([row])).hexdigest()
    return {"path": f"data/part-{number}.parquet", "rows": 1, "sha256": digest}, buffer.getvalue()


class Corpus:
    repo_id = "owner/corpus"

    def __init__(self):
        pairs = [make_shard("hello", 1), make_shard("<|endoftext|>", 2)]
        self.state = {"shards": [pair[0] for pair in pairs]}
        self.files = {pair[0]["path"]: pair[1] for pair in pairs}
        self.downloads = []

    def load(self):
        return copy.deepcopy(self.state), "pinned-source"

    def _download_bytes(self, path, revision):
        assert revision == "pinned-source"
        self.downloads.append(path)
        return self.files[path]


class Conflict(Exception):
    response = SimpleNamespace(status_code=409)


class Control:
    repo_id = "owner/control"

    def __init__(self):
        self.api = self
        self.files = {"state/progress.json": b"keep controller state"}
        self.sha = "first"
        self.fail_once = None
        self.writes = []
        self.conflict_status = 409

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.sha)

    def _download_bytes(self, path, revision):
        assert revision == self.sha
        if path not in self.files:
            raise EntryNotFoundError("missing")
        return self.files[path]

    def create_commit(self, operations, parent_commit, **kwargs):
        assert parent_commit == self.sha
        if self.fail_once:
            failure, self.fail_once = self.fail_once, None
            self.sha = "concurrent"
            if failure == "audit":
                self.files["tokens/ecb.json"] = b'{"other": true}'
            error = Conflict("parent conflict")
            error.response = SimpleNamespace(status_code=self.conflict_status)
            raise error
        self.writes.append({op.path_in_repo for op in operations})
        for op in operations:
            self.files[op.path_in_repo] = op.path_or_fileobj
        self.sha = f"commit-{len(self.writes)}"
        return SimpleNamespace(oid=self.sha)


def test_bounded_resume_avoids_recount_and_reports_partial_coverage():
    corpus, control = Corpus(), Control()
    first = run_audit("ecb", corpus, control, Encoding(), max_shards=1)
    assert first["counted_rows"] == 1
    assert first["exact_tokens_counted"] == 5
    assert first["total_rows"] == 2
    assert not first["complete_for_revision"]
    second = run_audit("ecb", corpus, control, Encoding(), max_shards=1)
    assert second["counted_rows"] == 2
    assert second["exact_tokens_counted"] == 18
    assert second["complete_for_revision"]
    assert len(corpus.downloads) == 2
    assert control.files["state/progress.json"] == b"keep controller state"
    assert all(paths == {"tokens/ecb.json"} for paths in control.writes)
    assert second["source_revision"] == "pinned-source"


def test_current_snapshot_excludes_removed_shard_cache():
    corpus, control = Corpus(), Control()
    run_audit("ecb", corpus, control, Encoding())
    corpus.state["shards"].pop()
    result = run_audit("ecb", corpus, control, Encoding())
    assert result["exact_tokens_counted"] == 5
    assert result["total_rows"] == 1
    assert len(corpus.downloads) == 2


def test_content_and_row_count_validation():
    shard, body = make_shard("hello", 1)
    with pytest.raises(ValueError, match="row count"):
        measure_shard(body, {**shard, "rows": 2}, Encoding())
    with pytest.raises(ValueError, match="hash"):
        measure_shard(body, {**shard, "sha256": "wrong"}, Encoding())


@pytest.mark.parametrize("status", [409, 412])
def test_safe_cas_rebase_only_unrelated_control_changes(monkeypatch, status):
    monkeypatch.setattr("token_audit.time.sleep", lambda _: None)
    control = Control()
    control.conflict_status = status
    cache = AuditCache(control, "ecb")
    assert cache.load() is None
    control.fail_once = "progress"
    cache.save({"hello": "audit"})
    assert json.loads(control.files["tokens/ecb.json"]) == {"hello": "audit"}
    control.fail_once = "audit"
    with pytest.raises(Conflict):
        cache.save({"hello": "replacement"})
    assert json.loads(control.files["tokens/ecb.json"]) == {"other": True}


def test_time_budget_can_stop_before_first_shard(monkeypatch):
    times = iter([0, 2])
    monkeypatch.setattr("token_audit.time.monotonic", lambda: next(times))
    corpus, control = Corpus(), Control()
    result = run_audit("ecb", corpus, control, Encoding(), work_seconds=1)
    assert result["counted_rows"] == 0
    assert result["total_rows"] == 2
    assert not result["complete_for_revision"]
    assert not corpus.downloads


def test_lost_commit_response_accepts_only_exact_persisted_bytes(monkeypatch):
    control = Control()
    cache = AuditCache(control, "ecb")
    cache.load()
    original = control.create_commit

    def lose_response(**kwargs):
        original(**kwargs)
        raise ConnectionError("response lost after commit")

    monkeypatch.setattr(control, "create_commit", lose_response)
    cache.save({"exact": 42})
    assert cache.revision == control.sha
    assert len(control.writes) == 1
    assert json.loads(control.files["tokens/ecb.json"]) == {"exact": 42}
