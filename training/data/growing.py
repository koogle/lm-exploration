"""Train from a raw corpus while it is still being prepared.

Preparation persists its SQLite database to the data volume every few minutes. Each
copy lists, in its `prepared_sources` table, the sources preparation has stopped
extending. When preparation is done it freezes a checksummed copy into the snapshot
folder and writes a completion record naming it. In the training function, a
`Refresher` thread reloads the volume and copies each newly persisted database to local
disk as an immutable, numbered generation; a Modal volume cannot reload while files on
it are open, so the loader never opens the volume copy. The loader's `GrowingDatabase`
reads the state file naming the newest local generation and whether preparation is
complete.

Documents are only ever appended, so a training cursor (pass, row ID, offset) stays
valid in every later generation. Preparation can later flag a stored document as a
cross-split duplicate, but never stores the duplicate itself; the growing loader
ignores those flags so every copy reads the same rows.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .subword import digest

# A snapshot whose database preparation is still appending to.
GROWING_FORMAT = "raw-text-growing-v1"
STATE_ENV = "GROWING_DATABASE_STATE"
# Written in the snapshot folder once preparation has persisted its final database.
COMPLETE_RECORD = "complete.json"
# How long the loader waits for newer prepared data before failing. Distributed
# collectives that wait on the loader need a longer timeout than this.
STALL_SECONDS = 3600
# Sources preparation has finished (it never extends them again), stored in the database.
FINISHED_TABLE = "prepared_sources"


class DataDeadline(Exception):
    """The session's time limit passed while waiting for newer prepared data."""


def finished_sources(db):
    """Sources a persisted copy marks as no longer growing."""
    try:
        return {row[0] for row in db.execute(f"SELECT source FROM {FINISHED_TABLE}")}
    except sqlite3.OperationalError:
        return set()


def mark_finished(db, sources):
    """Record `sources` as finished; the caller commits."""
    db.execute(f"CREATE TABLE IF NOT EXISTS {FINISHED_TABLE}(source TEXT PRIMARY KEY)")
    db.executemany(f"INSERT OR IGNORE INTO {FINISHED_TABLE} VALUES(?)", [(s,) for s in sources])


def max_persist_seconds():
    """The longest persist interval that still leaves a refresh inside the stall timeout."""
    return STALL_SECONDS // 2


class FixedDatabase:
    """A finished database: one complete copy that never waits."""

    def __init__(self, path):
        self.state = {"generation": 1, "path": str(path), "complete": True}
        self.deadline = None

    def current(self):
        return self.state

    def newer(self, generation):
        return self.state

    def interrupt(self):
        pass


def verified(folder, artifact, message):
    """The checksum-verified database an artifact entry names inside `folder`."""
    if Path(artifact["path"]).name != artifact["path"]:
        raise ValueError("Unsafe database path")
    path = Path(folder) / artifact["path"]
    if digest(path) != artifact["sha256"]:
        raise ValueError(message)
    return path


def database_copies(manifest_path, manifest):
    """Where a raw snapshot's loader finds its database.

    A frozen snapshot, or a growing one whose preparation completed, is one verified
    database. A growing snapshot still being prepared follows the refresher's state
    (see `serve`).
    """
    folder = Path(manifest_path).parent
    if manifest["format"] != GROWING_FORMAT:
        return FixedDatabase(
            verified(folder, manifest["database"], "Raw database checksum mismatch")
        )
    if STATE_ENV in os.environ:
        return GrowingDatabase(os.environ[STATE_ENV])
    complete = folder / COMPLETE_RECORD
    if not complete.exists():
        raise ValueError(f"A growing snapshot needs {STATE_ENV} until its preparation completes")
    frozen = json.loads(complete.read_text())["database"]
    return FixedDatabase(verified(folder, frozen, "Frozen growing database checksum mismatch"))


@contextmanager
def serve(snapshot, reload, interval_seconds=300.0):
    """Keep local copies of a snapshot's database while its preparation still runs.

    Sets `STATE_ENV` for the loader. A frozen or completed snapshot needs nothing.
    """
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["format"] != GROWING_FORMAT or (snapshot / COMPLETE_RECORD).exists():
        yield
        return
    database, complete = volume_paths(snapshot)
    local = tempfile.mkdtemp(prefix="growing-")
    refresher = Refresher(database, complete, local, reload, interval_seconds)
    os.environ[STATE_ENV] = str(refresher.state_path)
    try:
        # Later reloads start after one interval, once the caller has read the manifest.
        if not refresher.refresh():
            refresher.start()
        yield
    finally:
        refresher.stop.set()
        if refresher.is_alive():
            refresher.join()
        os.environ.pop(STATE_ENV, None)
        shutil.rmtree(local, ignore_errors=True)


def volume_paths(snapshot):
    """The persisted database and completion record of a growing snapshot folder.

    Snapshots live at `{version}/snapshots/{run_id}`; preparation persists the
    database at `{version}/raw.sqlite`.
    """
    snapshot = Path(snapshot)
    return snapshot.parents[1] / "raw.sqlite", snapshot / COMPLETE_RECORD


def write_json(path, value):
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(value) + "\n")
    os.replace(partial, path)


class GrowingDatabase:
    """The loader's view of the newest local copy of a database still being prepared."""

    def __init__(self, state_path, poll_seconds=10.0, stall_seconds=STALL_SECONDS):
        self.state_path = Path(state_path)
        self.poll_seconds = poll_seconds
        self.stall_seconds = stall_seconds
        # Wall-clock time (Unix seconds) at which a wait ends the session instead.
        self.deadline = None
        self.interrupted = threading.Event()

    def state(self):
        try:
            return json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return None

    def interrupt(self):
        """End current and later waits from another thread (e.g. a closing prefetcher)."""
        self.interrupted.set()

    def wait(self, condition):
        """Block until the state satisfies `condition`; fail after `stall_seconds`.

        Raises `DataDeadline` once the session deadline passes, so training can stop
        at its time limit with a checkpoint instead of being killed mid-wait.
        """
        started = time.monotonic()
        while True:
            state = self.state()
            if state is not None and condition(state):
                return state
            if self.deadline is not None and time.time() >= self.deadline:
                raise DataDeadline("Session time limit reached while waiting for prepared data")
            if time.monotonic() - started > self.stall_seconds:
                raise RuntimeError("Growing database stalled: no newer prepared data")
            if self.interrupted.wait(self.poll_seconds):
                raise RuntimeError("Growing database wait interrupted")

    def current(self):
        return self.wait(lambda state: True)

    def newer(self, generation):
        """The next generation, or the final one once preparation is complete."""
        return self.wait(lambda state: state["generation"] > generation or state["complete"])

    def complete(self):
        return self.wait(lambda state: state["complete"])


class Refresher(threading.Thread):
    """Copy each newly persisted volume database to local disk and publish its state.

    Once the completion record exists, the final copy comes from the frozen database it
    names, verified against its checksum.
    """

    def __init__(self, database, complete_record, local_dir, reload, interval_seconds=300.0):
        super().__init__(daemon=True, name="growing-database-refresher")
        self.database = Path(database)
        self.complete_record = Path(complete_record)
        self.local_dir = Path(local_dir)
        self.reload = reload
        self.interval_seconds = interval_seconds
        self.stop = threading.Event()
        self.generation = 0
        self.signature = None
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.local_dir / "state.json"

    def refresh(self):
        """One reload and copy; returns True once the final database is local."""
        try:
            self.reload()
        except Exception as exc:  # noqa: BLE001 -- "volume busy" is retried next interval
            print(
                json.dumps({"growing_reload_skipped": f"{type(exc).__name__}: {exc}"}), flush=True
            )
            return False
        complete = self.complete_record.exists()
        source, checksum = self.database, None
        if complete:
            frozen = json.loads(self.complete_record.read_text())["database"]
            if Path(frozen["path"]).name != frozen["path"]:
                raise ValueError("Unsafe frozen database path")
            source, checksum = self.complete_record.parent / frozen["path"], frozen["sha256"]
        stat = source.stat()
        signature = (str(source), stat.st_size, stat.st_mtime_ns)
        if signature != self.signature:
            target = self.local_dir / f"raw-{self.generation + 1:05d}.sqlite"
            partial = Path(str(target) + ".partial")
            try:
                shutil.copyfile(source, partial)
                if checksum is not None and digest(partial) != checksum:
                    raise ValueError("Frozen growing database checksum mismatch")
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            os.replace(partial, target)
            self.generation += 1
            self.signature = signature
        path = self.local_dir / f"raw-{self.generation:05d}.sqlite"
        # Publish the new copy before removing older ones, so the state never names a
        # deleted file. Open loader connections keep an unlinked older copy readable.
        write_json(
            self.state_path,
            {"generation": self.generation, "path": str(path), "complete": complete},
        )
        for old in self.local_dir.glob("raw-*.sqlite"):
            if old != path:
                old.unlink()
        return complete

    def run(self):
        """Refresh until the final database is local, starting after one interval.

        A failed refresh is retried at the next interval; one that keeps failing
        surfaces through the loader's stall timeout.
        """
        while not self.stop.wait(self.interval_seconds):
            try:
                if self.refresh():
                    return
            except Exception as exc:  # noqa: BLE001 -- e.g. a failed copy; retried next interval
                print(
                    json.dumps({"growing_refresh_failed": f"{type(exc).__name__}: {exc}"}),
                    flush=True,
                )
