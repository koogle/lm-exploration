import copy
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
