import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

from training_data.export_raw import complete_growing, export_growing
from training_data.growing import (
    DEADLINE_ENV,
    STATE_ENV,
    DataDeadline,
    Refresher,
    mark_finished,
    volume_paths,
    write_json,
)
from training_data.raw_token_stream import GROWING_FORMAT, RawTokenMixture
from training_data.subword import TOKENIZER_SHA256, digest, load_frozen

SCHEMA = """CREATE TABLE documents (
    hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
    provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0)"""


def insert(db, source, split, index):
    text = f"{source} {split} document {index} keeps its own words in order. " * 3
    count = len(load_frozen().encode(text, add_special_tokens=False).ids) + 1
    # Hash order differs from insertion order.
    key = hashlib.sha256(f"{index}".encode()).hexdigest()
    db.execute(
        "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
        (f"{key}-{split}", source, split, key, "{}", zlib.compress(text.encode()), count),
    )
    return count


def build(root, documents=12, snapshot_after=None):
    """A web corpus inserted one train and validation document at a time.

    Returns the database, the copy taken after `snapshot_after` documents and the
    per-split counts of the full database.
    """
    db = sqlite3.connect(root / "raw.sqlite")
    db.execute(SCHEMA)
    counts = {"train": 0, "validation": 0}
    prefix = None
    for index in range(documents):
        for split in counts:
            counts[split] += insert(db, "web", split, index)
        db.commit()
        if index + 1 == snapshot_after:
            prefix = root / "prefix.sqlite"
            db.execute("VACUUM INTO ?", (str(prefix),))
    db.close()
    return root / "raw.sqlite", prefix, counts


def freeze(database, complete):
    """Write the completion record naming a frozen copy of `database` beside it."""
    frozen = complete.parent / "frozen.sqlite"
    shutil.copyfile(database, frozen)
    write_json(complete, {"database": {"path": frozen.name, "sha256": digest(frozen)}})
    return frozen


def manifest(root, counts, growing=False):
    value = {
        "format": GROWING_FORMAT if growing else "raw-text-v1",
        "training_ready": True,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "sources": {
            "web": {
                "weight_percent": 100,
                "splits": {split: {"loss_tokens": count} for split, count in counts.items()},
            }
        },
    }
    if not growing:
        value["database"] = {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")}
    path = root / ("growing.json" if growing else "manifest.json")
    path.write_text(json.dumps(value))
    return path


def windows(mixture, limit=None):
    result = []
    while limit is None or len(result) < limit:
        try:
            result.append(mixture.next_block()["input_ids"])
        except StopIteration:
            break
    return result


class InsertionOrderTests(unittest.TestCase):
    def test_reads_insertion_order_and_resumes_mid_document(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, counts = build(root)
            path = manifest(root, counts)
            full = windows(RawTokenMixture(path, context=16))
            tokenizer = load_frozen()
            first = tokenizer.decode(full[0][1:])
            self.assertTrue(first.startswith("web train document 0 "), first)
            reader = RawTokenMixture(path, context=16)
            head = windows(reader, 5)
            state = reader.state_dict()
            self.assertGreater(state["raw_cursors"]["web"]["offset"], 0)
            resumed = RawTokenMixture(path, context=16)
            resumed.load_state_dict(json.loads(json.dumps(state)))
            self.assertEqual(head + windows(resumed), full)


class GrowingTests(unittest.TestCase):
    def test_growing_read_matches_full_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, prefix, counts = build(root, snapshot_after=4)
            full = windows(RawTokenMixture(manifest(root, counts), context=16))
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(prefix), "complete": False})
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
                head = windows(growing, 1)
                # The first pass catches up with the prefix and continues in the final copy.
                write_json(state, {"generation": 2, "path": str(database), "complete": True})
                self.assertEqual(head + windows(growing), full)

    def test_waits_for_newer_data_and_fails_when_stalled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, prefix, counts = build(root, snapshot_after=4)
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(prefix), "complete": False})
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
            growing.growing.poll_seconds, growing.growing.stall_seconds = 0.01, 0.05
            with self.assertRaisesRegex(RuntimeError, "stalled"):
                windows(growing)

    def test_later_passes_wait_for_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, counts = build(root, documents=2)
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(database), "complete": False})
            second_pass = {"web": {"pass": 1, "row": 0, "offset": 0}}
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
                saved = growing.state_dict()
                growing.load_state_dict({**saved, "raw_cursors": second_pass})
                growing.growing.poll_seconds, growing.growing.stall_seconds = 0.01, 0.05
                with self.assertRaisesRegex(RuntimeError, "stalled"):
                    windows(growing)
                write_json(state, {"generation": 1, "path": str(database), "complete": True})
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
                growing.load_state_dict({**saved, "raw_cursors": second_pass})
                self.assertEqual({block["pass"] for block in iter(growing.next_block, None)}, {2})

    def test_a_finished_source_reads_later_passes_before_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, counts = build(root, documents=2)
            db = sqlite3.connect(database)
            mark_finished(db, ["web"])
            db.close()
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(database), "complete": False})
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
                growing.growing.poll_seconds, growing.growing.stall_seconds = 0.01, 0.05
                passes = {block["pass"] for block in iter(growing.next_block, None)}
            self.assertEqual(passes, {1, 2})

    def test_reads_rows_flagged_after_they_were_stored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, counts = build(root)
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(database), "complete": True})
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                path = manifest(root, counts, growing=True)
                before = windows(RawTokenMixture(path, context=16, split="validation"))
                db = sqlite3.connect(database)
                db.execute("UPDATE documents SET conflict=1 WHERE rowid=2")
                db.commit()
                db.close()
                after = windows(RawTokenMixture(path, context=16, split="validation"))
            self.assertEqual(before, after)

    def test_a_wait_past_the_session_deadline_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, prefix, counts = build(root, snapshot_after=1)
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(prefix), "complete": False})
            deadline = str(time.time() + 0.2)
            with mock.patch.dict(os.environ, {STATE_ENV: str(state), DEADLINE_ENV: deadline}):
                growing = RawTokenMixture(manifest(root, counts, growing=True), context=16)
            growing.growing.poll_seconds = 0.01
            saved = growing.state_dict()
            with self.assertRaises(DataDeadline):
                growing.next_batch(64)
            self.assertEqual(growing.state_dict(), saved)

    def test_prefetcher_closes_while_waiting_for_newer_data(self):
        from training_data.prefetch import PrefetchedBatches

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, prefix, counts = build(root, snapshot_after=1)
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(prefix), "complete": False})
            with mock.patch.dict(os.environ, {STATE_ENV: str(state)}):
                path = manifest(root, counts, growing=True)
                # The prefix holds fewer windows than one batch, so the producer waits.
                prefetch = PrefetchedBatches(lambda: RawTokenMixture(path, context=16), 64)
                started = time.monotonic()
                prefetch.close()
                self.assertLess(time.monotonic() - started, 30)
                self.assertFalse(prefetch.thread.is_alive())

    def test_growing_snapshot_rejects_exact_source_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, counts = build(root, documents=2)
            path = manifest(root, counts, growing=True)
            value = json.loads(path.read_text())
            value["sources"]["web"]["target_tokens"] = 10
            path.write_text(json.dumps(value))
            state = root / "state.json"
            write_json(state, {"generation": 1, "path": str(database), "complete": True})
            with (
                mock.patch.dict(os.environ, {STATE_ENV: str(state)}),
                self.assertRaisesRegex(ValueError, "growing"),
            ):
                RawTokenMixture(path, context=16)


class RefresherTests(unittest.TestCase):
    def test_copies_each_new_database_and_stops_when_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, _ = build(root, documents=2)
            complete = root / "complete.json"
            reloads = []
            refresher = Refresher(database, complete, root / "local", lambda: reloads.append(1))
            self.assertFalse(refresher.refresh())
            state = json.loads(refresher.state_path.read_text())
            self.assertEqual((state["generation"], state["complete"]), (1, False))
            self.assertFalse(refresher.refresh())
            self.assertEqual(json.loads(refresher.state_path.read_text())["generation"], 1)
            db = sqlite3.connect(database)
            insert(db, "web", "train", 99)
            db.commit()
            db.close()
            frozen = freeze(database, complete)
            # Later preparation in the same version cannot change the final copy.
            db = sqlite3.connect(database)
            insert(db, "web", "train", 100)
            db.commit()
            db.close()
            self.assertTrue(refresher.refresh())
            state = json.loads(refresher.state_path.read_text())
            self.assertEqual((state["generation"], state["complete"]), (2, True))
            self.assertEqual(
                sorted(p.name for p in (root / "local").glob("raw-*.sqlite")),
                ["raw-00002.sqlite"],
            )
            self.assertEqual(Path(state["path"]).read_bytes(), frozen.read_bytes())
            self.assertEqual(len(reloads), 3)

    def test_rejects_a_frozen_copy_with_the_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, _ = build(root, documents=2)
            complete = root / "complete.json"
            freeze(database, complete)
            write_json(complete, {"database": {"path": "frozen.sqlite", "sha256": "0" * 64}})
            refresher = Refresher(database, complete, root / "local", lambda: None)
            with self.assertRaisesRegex(ValueError, "checksum"):
                refresher.refresh()
            self.assertEqual(list((root / "local").glob("raw-*")), [])

    def test_retries_a_failed_copy_without_leaving_partial_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database, _, _ = build(root, documents=2)
            complete = root / "complete.json"
            freeze(database, complete)
            refresher = Refresher(database, complete, root / "local", lambda: None, 0.01)
            copy = shutil.copyfile
            calls = []

            def flaky(source, target):
                calls.append(target)
                if len(calls) == 1:
                    Path(target).write_bytes(b"partial")
                    raise OSError("transient read error")
                return copy(source, target)

            with mock.patch("shutil.copyfile", side_effect=flaky):
                refresher.start()
                refresher.join(timeout=30)
            self.assertFalse(refresher.is_alive())
            state = json.loads(refresher.state_path.read_text())
            self.assertEqual((state["generation"], state["complete"]), (1, True))
            self.assertEqual(
                sorted(p.name for p in (root / "local").iterdir()),
                ["raw-00001.sqlite", "state.json"],
            )

    def test_skips_a_busy_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def busy():
                raise RuntimeError("files open")

            refresher = Refresher(root / "raw.sqlite", root / "complete.json", root / "local", busy)
            self.assertFalse(refresher.refresh())
            self.assertFalse(refresher.state_path.exists())


class ExportTests(unittest.TestCase):
    def test_export_and_complete_growing_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            version = Path(folder) / "v1"
            version.mkdir()
            database, _, counts = build(version, documents=2)
            recipe = {
                "sources": [{"name": "web", "weight_percent": 100, "repo": "r", "revision": "x"}]
            }
            db = sqlite3.connect(database)
            db.execute("CREATE TABLE settings(id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO settings VALUES(1, ?)", (json.dumps({"recipe": recipe}),))
            db.commit()
            db.close()
            snapshot = version / "snapshots" / "run"
            written = export_growing(version, snapshot, ["audit"])
            self.assertEqual(written["format"], GROWING_FORMAT)
            self.assertEqual(
                written["sources"]["web"]["splits"]["train"]["loss_tokens"], counts["train"]
            )
            self.assertEqual(volume_paths(snapshot), (database, snapshot / "complete.json"))
            with self.assertRaises(ValueError):
                export_growing(version, snapshot, [])
            record = complete_growing(version, snapshot, capacity_verified_for_tokens=5)
            self.assertEqual(json.loads((snapshot / "complete.json").read_text()), record)
            frozen = snapshot / record["database"]["path"]
            self.assertEqual(digest(frozen), record["database"]["sha256"])
            rows = "SELECT hash, split FROM documents ORDER BY rowid"
            with sqlite3.connect(frozen) as copy, sqlite3.connect(database) as live:
                self.assertEqual(copy.execute(rows).fetchall(), live.execute(rows).fetchall())
            shutil.rmtree(snapshot)


if __name__ == "__main__":
    unittest.main()
