"""Pinned artifact downloads verify bytes before use and never require Git copies."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication.artifacts import fetch_artifacts, read_index, verify


class ResultArtifactTests(unittest.TestCase):
    def test_pinning_and_corrupt_download_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "remote.json"
            source.write_bytes(b"{}\n")
            entry = {
                "path": "reports/result.json",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": 3,
            }
            index = {
                "repo_id": "example/results",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "files": {"result.json": entry},
            }
            path = folder / "index.json"
            path.write_text(json.dumps(index))
            with patch("huggingface_hub.hf_hub_download", return_value=str(source)) as download:
                fetch_artifacts(folder / "output", path)
                self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
                self.assertEqual(download.call_args.kwargs["repo_type"], "dataset")
                source.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch_artifacts(folder / "output", path)
                verify(folder / "output/result.json", entry)
            path.write_text(json.dumps({**index, "revision": "main"}))
            with self.assertRaisesRegex(ValueError, "immutable dataset revision"):
                read_index(path)
