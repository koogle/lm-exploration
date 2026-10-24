import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data.cached_shards import CachedShards
from training_data.export_raw import export_raw
from training_data.prepared_snapshot import find_prepared_snapshot
from training_data.raw_corpus import RawCorpus


class PreparedSnapshotTests(unittest.TestCase):
    def test_reuse_checks_identity_capacity_and_integrity(self):
        recipe = {
            "sources": [
                {
                    "name": "fineweb_edu",
                    "weight_percent": 100,
                    "files": ["fixture"],
                    "repo": "test",
                    "revision": "0" * 40,
                }
            ]
        }
        rows = [{"id": str(i), "text": f"Document {i} has distinct text."} for i in range(100)]

        def reader(source, shard, start):
            yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertIsNone(find_prepared_snapshot(folder, recipe, 100))
            corpus = RawCorpus(folder, recipe)
            corpus.prepare("fineweb_edu", reader=reader)
            corpus.close()
            snapshot = folder / "snapshots" / "first"
            export_raw(folder, snapshot)
            path = snapshot / "manifest.json"
            original = path.read_bytes()
            self.assertEqual(find_prepared_snapshot(folder, recipe, 100), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertIsNone(find_prepared_snapshot(folder, recipe, 10**9))
            changed = copy.deepcopy(recipe)
            changed["sources"][0]["revision"] = "1" * 40
            self.assertIsNone(find_prepared_snapshot(folder, changed, 100))
            with (snapshot / "raw.sqlite").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                find_prepared_snapshot(folder, recipe, 100)


class CachedShardsTests(unittest.TestCase):
    def test_local_shard_resume_and_old_shard_eviction(self):
        source = {"name": "code_python", "format": "json.gz", "repo": "test", "revision": "0" * 40}
        with tempfile.TemporaryDirectory() as temporary:

            def download(**kwargs):
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with gzip.open(path, "wt") as handle:
                        for i in range(3):
                            handle.write(json.dumps({"row": i}) + "\n")
                return str(path)

            reader = CachedShards(temporary)
            with patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch:
                self.assertEqual(
                    list(reader(source, "a.gz", start=1)), [(1, {"row": 1}), (2, {"row": 2})]
                )
                self.assertTrue((Path(temporary) / "code_python/a.gz").exists())
                list(reader(source, "b.gz"))
                self.assertFalse((Path(temporary) / "code_python/a.gz").exists())
                self.assertTrue((Path(temporary) / "code_python/b.gz").exists())
                self.assertEqual(fetch.call_args.kwargs["revision"], source["revision"])

    def test_parquet_shards_download_whole_and_resume_mid_shard(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = {"name": "web", "format": "parquet", "repo": "test", "revision": "0" * 40}
        with tempfile.TemporaryDirectory() as temporary:

            def download(**kwargs):
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.table({"text": [f"row {i}" for i in range(2500)]})
                pq.write_table(table, path, row_group_size=1000)
                return str(path)

            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                rows = list(CachedShards(temporary)(source, "s.parquet", start=1999))
            self.assertEqual(rows[0], (1999, {"text": "row 1999"}))
            self.assertEqual([i for i, _ in rows], list(range(1999, 2500)))


class PreparationReuseTests(unittest.TestCase):
    def test_reuse_survives_code_edits_and_honors_pass_caps(self):
        recipe = {
            "sources": [
                {
                    "name": "small",
                    "weight_percent": 100,
                    "files": ["fixture"],
                    "repo": "test",
                    "revision": "0" * 40,
                    "max_passes": 4,
                }
            ]
        }
        rows = [{"id": str(i), "text": f"Document {i} has distinct text."} for i in range(100)]

        def reader(source, shard, start):
            yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            corpus = RawCorpus(folder, recipe)
            corpus.prepare("small", reader=reader)
            train = corpus.counts("small")["train"]
            corpus.close()
            export_raw(folder, folder / "snapshots" / "first")
            # The identity names data inputs and a preparation version, not code hashes.
            manifest = json.loads((folder / "snapshots/first/manifest.json").read_text())
            self.assertEqual(
                set(manifest["staging_identity"]), {"recipe", "tokenizer", "split", "preparation"}
            )
            # Four passes stretch the same unique tokens to four times the target.
            self.assertIsNotNone(find_prepared_snapshot(folder, recipe, train * 4))
            self.assertIsNone(find_prepared_snapshot(folder, recipe, train * 4 + 1))


class PersistTests(unittest.TestCase):
    def test_persist_writes_a_consistent_copy(self):
        import sqlite3

        from training_data.fast_raw_corpus import FastRawCorpus

        recipe = {
            "sources": [
                {
                    "name": "web",
                    "weight_percent": 100,
                    "files": ["fixture"],
                    "repo": "test",
                    "revision": "0" * 40,
                }
            ]
        }
        rows = [{"id": str(i), "text": f"Document {i} has distinct text."} for i in range(50)]

        def reader(source, shard, start):
            yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = FastRawCorpus(root / "local", recipe, reader=reader)
            self.assertTrue(corpus.prepare_parallel(100, max_seconds=5)["ready"])
            corpus.persist(root / "volume/raw.sqlite")
            expected = corpus.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            corpus.close()
            copy_db = sqlite3.connect(root / "volume/raw.sqlite")
            self.assertEqual(
                copy_db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], expected
            )
            copy_db.close()
            self.assertFalse((root / "volume/raw.sqlite.partial").exists())
