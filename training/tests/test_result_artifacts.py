"""Generated PR13 evidence lives on a pinned dataset, not in the source tree."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication.artifacts import artifact_path, fetch_artifacts, read_index

ROOT = Path(__file__).resolve().parents[1]


class ResultArtifactTests(unittest.TestCase):
    def fixture(self, folder):
        data = b'{"loss": 2.7}\n'
        source = folder / "remote.json"
        source.write_bytes(data)
        index = {
            "repo_id": "example/results",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "files": {
                "results/test.json": {
                    "path": "pr-13/training/results/test.json",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            },
        }
        index_path = folder / "index.json"
        index_path.write_text(json.dumps(index))
        return source, index, index_path

    def test_pinned_download_and_explicit_verified_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, index, index_path = self.fixture(folder)
            output = folder / "download"
            with patch("huggingface_hub.hf_hub_download", return_value=str(source)) as download:
                fetch_artifacts(output, index_path)
                download.assert_called_once_with(
                    repo_id="example/results",
                    repo_type="dataset",
                    revision="a" * 40,
                    filename="pr-13/training/results/test.json",
                )
            with patch("publication.artifacts.read_index", return_value=index):
                target = artifact_path(output, "results/test.json")
                self.assertEqual(target.read_bytes(), source.read_bytes())
                target.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    artifact_path(output, "results/test.json")

    def test_corrupt_download_is_not_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, _, index_path = self.fixture(folder)
            source.write_bytes(b"wrong bytes")
            with (
                patch("huggingface_hub.hf_hub_download", return_value=str(source)),
                self.assertRaisesRegex(ValueError, "checksum mismatch"),
            ):
                fetch_artifacts(folder / "output", index_path)
            self.assertFalse((folder / "output/results/test.json").exists())

    def test_mutable_revisions_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, index, index_path = self.fixture(folder)
            index_path.write_text(json.dumps({**index, "revision": "main"}))
            with self.assertRaisesRegex(ValueError, "immutable dataset revision"):
                read_index(index_path)
            index["files"]["../escape"] = next(iter(index["files"].values()))
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                read_index(index_path)

    def test_migrated_artifacts_are_not_in_the_code_tree(self):
        index = read_index()
        self.assertEqual(index["repo_type"], "dataset")
        self.assertEqual(len(index["files"]), 19)
        self.assertEqual(sum(e["size_bytes"] for e in index["files"].values()), 170805)
        for name in index["files"]:
            with self.subTest(artifact=name):
                self.assertFalse((ROOT / name).exists())
