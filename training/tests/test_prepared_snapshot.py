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
