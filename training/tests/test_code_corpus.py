import gzip
import hashlib
import json
import random
import shutil
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from training_data.lexical_screen import LexicalScreen, normalized_hash, shingles
from training_data.multilingual_code import SCHEMA, select
from training_data.parallel_code import (
    build_exclusions,
    merge_partitions,
    partition_shards,
    prepare_partition,
)
from training_data.readiness_audit import audit


def add(db, text, split, count, parent="code_python:repo/project", source="code"):
    key = hashlib.sha256(text.encode()).hexdigest()
    db.execute(
        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
        (
            key,
            source,
            split,
            parent,
            json.dumps({"language": "python", "task_family": "tests"}),
            zlib.compress(text.encode()),
            count,
        ),
    )
    return key


def code_row(index, repository="repo/train", **changes):
    return {
        "code": f"def function_{index}(argument):\n    # A unique small example.\n"
        f"    return argument + {index}\n",
        "repo_name": repository,
        "path": f"tests/test_{index}.py",
        "language": "Python",
        "license": "mit",
        **changes,
    }


class Encoding:
    def __init__(self, text):
        self.ids = [4] * len(text.split())


class WordTokenizer:
    def encode(self, text, **kwargs):
        return Encoding(text)


class CodeCollectionTests(unittest.TestCase):
    def test_row_selection(self):
        kept = select(code_row(1))
        self.assertEqual(
            (kept["language"], kept["task_family"], kept["parent"]),
            ("python", "tests", "code_python:repo/train"),
        )
        rejected = [
            code_row(1, license="gpl-3.0"),
            code_row(1, language="COBOL"),
            code_row(1, path="node_modules/x/index.js"),
            code_row(1, code="x = 1"),
            code_row(1, code="# Automatically generated, do not edit\n" + "x = 1\n" * 20),
        ]
        for row in rejected:
            with self.subTest(row=row):
                self.assertIsNone(select(row))

    def test_disjoint_partitions_cover_remaining_shards(self):
        files = [f"{i}.parquet" for i in range(31)]
        groups = [partition_shards(files, 9, w, 8) for w in range(8)]
        indices = [i for group in groups for i, _ in group]
        self.assertEqual(sorted(indices), list(range(9, 31)))
        self.assertEqual(len(set(indices)), 22)

    def test_exclusions_keep_prior_hashes_code_norms_and_heldout_repositories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior.sqlite"
            db = sqlite3.connect(prior)
            db.executescript(SCHEMA)
            add(db, "old heldout code", "validation", 10, parent="code_python:HELD/repo")
            add(db, "old train code", "train", 10, parent="code_python:held/repo")
            add(db, "x  =  1  # web text", "train", 10, parent="web:page", source="web")
            db.commit()
            db.close()
            identity = build_exclusions([prior], root / "base", files=["a.parquet", "b.parquet"])
            self.assertEqual(sorted(identity["files"]), ["a.parquet", "b.parquet"])
            db = sqlite3.connect(root / "base/exclusions.sqlite")
            split = db.execute(
                "SELECT split FROM repositories WHERE parent='code_python:held/repo'"
            )
            self.assertEqual(split.fetchone()[0], "heldout")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM hashes").fetchone()[0], 3)
            norms = {row[0] for row in db.execute("SELECT hash FROM normalized")}
            self.assertEqual(
                norms, {normalized_hash("old heldout code"), normalized_hash("old train code")}
            )
            db.close()
            with self.assertRaisesRegex(ValueError, "already exist"):
                build_exclusions([prior], root / "base", files=[])

    def test_partition_resumes_without_reusing_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior.sqlite"
            db = sqlite3.connect(prior)
            db.executescript(SCHEMA)
            add(db, "old train code", "train", 10, parent="code_python:repo/train")
            db.commit()
            db.close()
            files = [f"shard-{i}.parquet" for i in range(4)]
            for i, name in enumerate(files):
                rows = [code_row(i * 10 + j) for j in range(4)]
                pq.write_table(pa.Table.from_pylist(rows), root / name)
            identity = build_exclusions([prior], root / "base", files=files)

            def download(repo, name, **kwargs):
                destination = Path(kwargs["local_dir"]) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
                return str(destination)

            output = root / "worker"
            exclusions = root / "base/exclusions.sqlite"
            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                first = prepare_partition(output, exclusions, identity, 0, 2, 1)
                second = prepare_partition(
                    output, exclusions, identity, 0, 2, first["fresh_code_tokens"] + 1
                )
            self.assertGreater(second["fresh_code_tokens"], first["fresh_code_tokens"])
            db = sqlite3.connect(output / "raw.sqlite")
            # Worker 0 owns two of the four shards; resume added the second without repeats.
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 8)
            splits = {row[0] for row in db.execute("SELECT DISTINCT split FROM documents")}
            self.assertEqual(splits, {"train"})  # A prior train repository stays train.
            db.close()

    def test_merge_dedups_globally_marks_cross_split_conflicts_and_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                source.executescript(
                    SCHEMA
                    + "CREATE TABLE dedup(normalized TEXT PRIMARY KEY,hash TEXT UNIQUE,split TEXT);"
                    + "CREATE TABLE settings(id INTEGER,value TEXT);"
                    + "CREATE TABLE cursor(id INTEGER,shard INTEGER,row INTEGER);"
                )
                source.execute("INSERT INTO settings VALUES (1,?)", (json.dumps({"worker": i}),))
                source.execute("INSERT INTO cursor VALUES (1,?,100)", (i,))
                for text, split, count in rows:
                    key = add(source, text, split, count, parent=f"code_python:repo/worker{i}")
                    source.execute(
                        "INSERT INTO dedup VALUES (?,?,?)", (normalized_hash(text), key, split)
                    )
                source.commit()
                source.close()
            central = root / "central.sqlite"
            result = merge_partitions(central, paths)
            # The validation copy of "add" conflicts with its train copy, leaving only "sub".
            self.assertEqual(result["fresh_code_tokens"], 20)
            again = merge_partitions(central, paths)
            self.assertEqual(again["fresh_code_tokens"], 20)
            self.assertEqual(again["added_unique_train_tokens"], 0)


class LexicalScreenTests(unittest.TestCase):
    def test_matches_any_shared_13_word_shingle(self):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen"
        screen = LexicalScreen(shingles(text))
        self.assertTrue(screen.matches("prefix " + text.upper() + " suffix"))
        self.assertTrue(screen.matches(text.replace(" ", "\n! ")))
        self.assertFalse(screen.matches(text + "extra"))
        words = text.split()
        random.Random(42).shuffle(words)
        self.assertFalse(screen.matches(" ".join(words)))
        self.assertEqual(normalized_hash("a = 123"), normalized_hash("a=123\n"))


class ReadinessAuditTests(unittest.TestCase):
    def test_dedup_parent_leakage_benchmarks_capacity_and_heldouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite"
            db = sqlite3.connect(source)
            db.executescript(SCHEMA)
            add(db, "a = 123", "train", 10, "code_python:Owner/repo", "code_python")
            add(db, "a=123", "train", 10, "code_python:owner/clone", "code_python")
            add(db, "validation body", "validation", 20, "code_python:HELD/repo", "code_python")
            add(db, "leaked train body", "train", 30, "code:held/repo", "code")
            benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
            add(db, benchmark, "train", 40, "code_python:benchmark/repo", "code_python")
            add(db, "a second document", "train", 50, "code_python:benchmark/repo", "code_python")
            add(db, "clean validation example", "validation", 60, "code_python:clean/held")
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                ("test-only-hash", "code_python", "test", "code_python:private/test", "{}", b"", 0),
            )
            db.commit()
            db.close()
            with patch("training_data.readiness_audit.load_frozen", return_value=WordTokenizer()):
                result = audit(
                    [("prior", source, False)], root / "audit", set(shingles(benchmark)), []
                )
            # Only one of the two normalized copies of "a = 123" remains; the held-out
            # repository (across code sources and case) and the benchmark repo are blocked.
            self.assertEqual(result["capacity"][0]["unique_train_tokens"], 10)
            self.assertEqual(result["benchmark_hit_documents"], 1)
            self.assertEqual(result["blocked_parents"], 2)
            self.assertEqual(result["heldout"]["code/python/tests"]["parents"], 1)
            with gzip.open(root / "audit/heldout-windows.jsonl.gz", "rt") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["source"] for row in rows], ["code"])
            self.assertEqual(rows[0]["tokens"][0], 1)
            with self.assertRaisesRegex(ValueError, "exists"):
                audit([("prior", source, False)], root / "audit", set(), [])
            # Resume reuses scanned metadata; dropping the benchmark screen unblocks its repo.
            with patch("training_data.readiness_audit.load_frozen", return_value=WordTokenizer()):
                resumed = audit(
                    [("prior", source, False)],
                    root / "audit",
                    set(),
                    [],
                    benchmark_screen=False,
                    tokenizer_diagnostics=False,
                    resume=True,
                )
            self.assertEqual(resumed["capacity"][0]["unique_train_tokens"], 100)
            self.assertEqual(resumed["blocked_parents"], 1)
            self.assertEqual(resumed["documents_scanned"], result["documents_scanned"])
            self.assertEqual(resumed["tokenizer"], {})
            self.assertIsNone(resumed["benchmark_hit_documents"])


if __name__ == "__main__":
    unittest.main()
