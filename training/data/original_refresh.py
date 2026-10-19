"""Fork the old staging cursor while marking every previous document ineligible as fresh."""

import json
import sqlite3
from pathlib import Path


def fork_original(folder, original_db, eu_db):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "raw.sqlite"
    if not final.exists():
        temporary = folder / "fork.sqlite"
        temporary.unlink(missing_ok=True)
        with sqlite3.connect(original_db) as source, sqlite3.connect(temporary) as target:
            source.backup(target)
            target.executescript("""DROP TRIGGER IF EXISTS fresh_insert; DROP TRIGGER IF EXISTS fresh_conflict;
                DROP TABLE IF EXISTS fresh_counts; DROP TABLE IF EXISTS excluded_prior;
                DROP TABLE IF EXISTS allocation_policy; DROP TABLE IF EXISTS fresh_identity;
                CREATE TABLE excluded_prior (hash TEXT PRIMARY KEY); INSERT INTO excluded_prior SELECT hash FROM documents;""")
            target.execute("ATTACH DATABASE ? AS eu", (str(eu_db),))
            target.execute("INSERT OR IGNORE INTO excluded_prior SELECT hash FROM eu.documents")
            target.commit()
        temporary.replace(final)
    with sqlite3.connect(final) as db:
        return json.loads(db.execute("SELECT value FROM settings WHERE id=1").fetchone()[0])[
            "recipe"
        ]
