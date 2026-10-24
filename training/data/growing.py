"""Train from a raw corpus while it is still being prepared.

Preparation persists its SQLite database to the data volume every few minutes and
writes a completion record when it is done. In the training function, a `Refresher`
thread reloads the volume and copies each newly persisted database to local disk as an
immutable, numbered generation; a Modal volume cannot reload while files on it are open,
so the loader never opens the volume copy. The loader's `GrowingDatabase` reads the
state file naming the newest local generation and whether preparation is complete.

Documents are only ever appended, so a training cursor (pass, row ID, offset) stays
valid in every later generation.
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path

STATE_ENV = "GROWING_DATABASE_STATE"
# Written in the snapshot folder once preparation has persisted its final database.
COMPLETE_RECORD = "complete.json"
# How long the loader waits for newer prepared data before failing. Distributed
# collectives that wait on the loader need a longer timeout than this.
STALL_SECONDS = 3600


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

    def state(self):
        try:
            return json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return None

    def wait(self, condition):
        """Block until the state satisfies `condition`; fail after `stall_seconds`."""
        started = time.monotonic()
        while True:
            state = self.state()
            if state is not None and condition(state):
                return state
            if time.monotonic() - started > self.stall_seconds:
                raise RuntimeError("Growing database stalled: no newer prepared data")
            time.sleep(self.poll_seconds)

    def current(self):
        return self.wait(lambda state: True)

    def newer(self, generation):
        """The next generation, or the final one once preparation is complete."""
        return self.wait(lambda state: state["generation"] > generation or state["complete"])

    def complete(self):
        return self.wait(lambda state: state["complete"])


class Refresher(threading.Thread):
    """Copy each newly persisted volume database to local disk and publish its state."""

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
        self.error = None
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
        stat = self.database.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature != self.signature:
            self.generation += 1
            target = self.local_dir / f"raw-{self.generation:05d}.sqlite"
            partial = Path(str(target) + ".partial")
            shutil.copyfile(self.database, partial)
            os.replace(partial, target)
            self.signature = signature
            # Open loader connections keep an unlinked older copy readable.
            for old in self.local_dir.glob("raw-*.sqlite"):
                if old != target:
                    old.unlink()
        path = self.local_dir / f"raw-{self.generation:05d}.sqlite"
        write_json(
            self.state_path,
            {"generation": self.generation, "path": str(path), "complete": complete},
        )
        return complete

    def run(self):
        """Refresh until the final database is local, starting after one interval."""
        try:
            while not self.stop.wait(self.interval_seconds):
                if self.refresh():
                    return
        except Exception as exc:  # noqa: BLE001 -- surfaced through the loader's stall timeout
            self.error = exc
            print(
                json.dumps({"growing_refresher_failed": f"{type(exc).__name__}: {exc}"}), flush=True
            )
