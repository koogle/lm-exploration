import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data.hf_patches import patch_record, prepare


def row(value=1):
    return {
        "repository": "Owner/Project",
        "sha": str(value),
        "repository_license": "MIT",
        "patch": f"diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-print(0)\n+print({value})\n",
    }


class PatchTests(unittest.TestCase):
    def test_related_versions_allowed_and_repo_holdout_protected(self):
        args = ("private/code", "pin", "changes/a.jsonl.gz", 0)
        a = patch_record(row(1), *args, {"code_python:owner/project": "train"})
        b = patch_record(row(2), *args, {"code_python:owner/project": "train"})
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1:3], b[1:3])
        self.assertTrue(a[3]["related_version_overlap_allowed"])
        self.assertIsNone(patch_record(row(), *args, {"code_python:owner/project": "heldout"}))
        unsupported = dict(row(), repository_license="NOASSERTION")
        self.assertIsNone(patch_record(unsupported, *args, {}))

    def test_ingest_exact_dedup_resume_and_frozen_token_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = sqlite3.connect(root / "exclusions.sqlite")
            db.execute("CREATE TABLE repositories(parent TEXT,split TEXT)")
            db.execute("INSERT INTO repositories VALUES('code_python:owner/project','train')")
            db.commit()
            db.close()
            shard = root / "shard.gz"
            with gzip.open(shard, "wt") as out:
                for item in [row(1), row(1), row(2)]:
                    out.write(json.dumps(item) + "\n")
            datasets = [{"repo": "private/code", "revision": "pin"}]
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.hf_hub_download", return_value=str(shard)) as download,
            ):
                api.return_value.list_repo_files.return_value = ["changes/a.jsonl.gz"]
                first = prepare(root / "output", datasets, root / "exclusions.sqlite")
                second = prepare(root / "output", datasets, root / "exclusions.sqlite")
            self.assertEqual(first, second)
            self.assertEqual(first["splits"]["train"]["documents"], 2)
            self.assertGreater(first["splits"]["train"]["tokens"], 10)
            self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
