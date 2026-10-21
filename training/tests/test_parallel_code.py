import hashlib
import json
import random
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from training_data.lexical_screen import LexicalScreen
from training_data.multilingual_code import SCHEMA
from training_data.parallel_code import build_exclusions, merge_partitions, partition_shards
from training_data.parallel_code_merge import merge_partitions as fast_merge
from training_data.readiness_audit import normalized_hash, shingles


def add(db, text, split, count, parent="code_python:repo/project"):
    key = hashlib.sha256(text.encode()).hexdigest()
    db.execute(
        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
        (
            key,
            "code",
            split,
            parent,
            json.dumps({"language": "python", "task_family": "tests"}),
            zlib.compress(text.encode()),
            count,
        ),
    )
    return key


class ParallelCodeTests(unittest.TestCase):
    def test_disjoint_partition_union_after_partial_shard(self):
        files = [f"{i}.parquet" for i in range(31)]
        groups = [partition_shards(files, 9, w, 8) for w in range(8)]
        self.assertEqual(sorted(i for group in groups for i, p in group), list(range(9, 31)))
        self.assertEqual(len({i for group in groups for i, p in group}), 22)

    def test_merge_global_dedup_cross_split_conflict_and_replay(self):
        for engine in (merge_partitions, fast_merge):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                central = root / "central.sqlite"
                db = sqlite3.connect(central)
                db.executescript(SCHEMA)
                db.executescript(
                    "CREATE TABLE normalized(hash TEXT PRIMARY KEY); CREATE TABLE stats(kind TEXT,name TEXT,tokens INTEGER,documents INTEGER,PRIMARY KEY(kind,name));"
                )
                text = "print(100)"
                add(db, text, "train", 5)
                db.execute("INSERT INTO normalized VALUES (?)", (normalized_hash(text),))
                db.commit()
                db.close()
                paths = []
                for i, rows in enumerate(
                    [
                        [("def add(a,b): return a+b", "train", 10)],
                        [
                            ("def add(a, b): return a + b", "validation", 10),
                            ("def sub(a,b): return a-b", "train", 20),
                        ],
                    ]
                ):
                    path = root / f"worker{i}.sqlite"
                    paths.append(path)
                    source = sqlite3.connect(path)
                    source.executescript(SCHEMA)
                    source.executescript(
                        "CREATE TABLE dedup(normalized TEXT PRIMARY KEY,hash TEXT UNIQUE,split TEXT); CREATE TABLE settings(id INTEGER,value TEXT); CREATE TABLE cursor(id INTEGER,shard INTEGER,row INTEGER);"
                    )
                    source.execute(
                        "INSERT INTO settings VALUES (1,?)", (json.dumps({"worker": i}),)
                    )
                    source.execute("INSERT INTO cursor VALUES (1,?,100)", (i,))
                    for text, split, count in rows:
                        key = add(source, text, split, count, parent=f"code_python:repo/worker{i}")
                        source.execute(
                            "INSERT INTO dedup VALUES (?,?,?)", (normalized_hash(text), key, split)
                        )
                    source.commit()
                    source.close()
                result = engine(central, paths)
                self.assertEqual(result["fresh_code_tokens"], 25)
                self.assertEqual(result["added_unique_train_tokens"], 20)
                again = engine(central, paths)
                self.assertEqual(again["fresh_code_tokens"], 25)
                self.assertEqual(again["added_unique_train_tokens"], 0)

    def test_snapshot_exclusions_keep_hashes_and_parent_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            central = root / "central.sqlite"
            prior = root / "prior.sqlite"
            db = sqlite3.connect(central)
            db.executescript(SCHEMA)
            db.executescript(
                "CREATE TABLE prior_hashes(hash TEXT PRIMARY KEY); CREATE TABLE normalized(hash TEXT PRIMARY KEY); CREATE TABLE cursor(id INTEGER,shard INTEGER,row INTEGER);"
            )
            db.execute("INSERT INTO prior_hashes VALUES ('prior-hash')")
            db.execute("INSERT INTO normalized VALUES ('old-normalized')")
            db.execute("INSERT INTO cursor VALUES (1,9,1234)")
            add(db, "old serial code", "train", 10)
            db.commit()
            db.close()
            db = sqlite3.connect(prior)
            db.executescript(SCHEMA)
            key = add(db, "old heldout code", "validation", 10, parent="code_python:HELD/repo")
            db.execute("UPDATE documents SET source='code_python' WHERE hash=?", (key,))
            db.commit()
            db.close()
            metadata = build_exclusions(
                central, prior, root / "base", files=["a.parquet", "b.parquet"]
            )
            self.assertEqual((metadata["start_shard"], metadata["start_row"]), (9, 1234))
            db = sqlite3.connect(root / "base/exclusions.sqlite")
            self.assertEqual(
                db.execute(
                    "SELECT split FROM repositories WHERE parent='code_python:held/repo'"
                ).fetchone()[0],
                "heldout",
            )
            self.assertEqual(db.execute("SELECT COUNT(*) FROM hashes").fetchone()[0], 2)
            db.close()

    def test_compiled_lexical_match_agrees_with_reference_and_word_boundaries(self):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen"
        fingerprints = set(shingles(text))
        screen = LexicalScreen(fingerprints)
        samples = [
            text,
            text.upper(),
            "prefix " + text + " suffix",
            text + "extra",
            "x" + text,
            text.replace(" ", "\n! "),
            "one two",
            "één " + text + " åäö",
        ]
        rng = random.Random(42)
        for _ in range(40):
            words = text.split()
            rng.shuffle(words)
            samples.append(" ".join(words))
        for sample in samples:
            self.assertEqual(
                screen.matches(sample), any(s in fingerprints for s in shingles(sample)), sample
            )

    def test_partition_tokenization_resumes_without_reusing_rows(self):
        import shutil
        from unittest.mock import patch

        import pyarrow as pa
        import pyarrow.parquet as pq
        from training_data.parallel_code import prepare_partition
        from training_data.subword import digest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "exclusions.sqlite"
            db = sqlite3.connect(old)
            db.executescript(
                "CREATE TABLE hashes(hash TEXT PRIMARY KEY); CREATE TABLE normalized(hash TEXT PRIMARY KEY); CREATE TABLE repositories(parent TEXT PRIMARY KEY,split TEXT);"
            )
            db.execute("INSERT INTO repositories VALUES ('code_python:repo/train','train')")
            db.commit()
            db.close()
            files = [f"shard-{i}.parquet" for i in range(4)]
            for i, file in enumerate(files):
                rows = [
                    {
                        "code": f"def function_{i}_{j}(argument):\n    # A unique small example for this fixture.\n    return argument + {i * 10 + j}\n",
                        "repo_name": "repo/train",
                        "path": f"tests/test_{i}_{j}.py",
                        "language": "Python",
                        "license": "mit",
                    }
                    for j in range(4)
                ]
                pq.write_table(pa.Table.from_pylist(rows), root / file)
            identity = {
                "files": files,
                "start_shard": 0,
                "start_row": 2,
                "exclusions_sha256": digest(old),
            }

            def download(repo, file, **kwargs):
                destination = Path(kwargs["local_dir"]) / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / file, destination)
                return str(destination)

            output = root / "worker"
            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                first = prepare_partition(output, old, identity, 0, 2, 1)
                db = sqlite3.connect(output / "raw.sqlite")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 2)
                db.close()
                second = prepare_partition(
                    output, old, identity, 0, 2, first["fresh_code_tokens"] + 1
                )
            self.assertGreater(second["fresh_code_tokens"], first["fresh_code_tokens"])
            db = sqlite3.connect(output / "raw.sqlite")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 6)
            db.close()
