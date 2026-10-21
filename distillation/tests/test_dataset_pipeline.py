import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from distillation.dataset_archive import archive
from distillation.dataset_pipeline import phase_goal, publish_chunks, teacher_count
from distillation.scale_data import atomic_json, available_chunks


class DatasetPipelineTests(unittest.TestCase):
    def test_phase_boundaries_preserve_total_budget(self):
        self.assertEqual(
            phase_goal(61_646_516, 4_500_000_000, 1_000_000_000), 1_000_000_000
        )
        self.assertEqual(
            phase_goal(1_000_000_000, 4_500_000_000, 1_000_000_000), 2_000_000_000
        )
        self.assertEqual(
            phase_goal(4_000_000_000, 4_500_000_000, 1_000_000_000), 4_500_000_000
        )
        self.assertEqual(
            phase_goal(4_500_000_000, 4_500_000_000, 1_000_000_000), 4_500_000_000
        )

    def test_sizing_and_dynamic_publish_preserve_prefix(self):
        self.assertEqual(teacher_count(100000, 2500), 50)
        with self.assertRaises(ValueError):
            teacher_count(0, 2500)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for worker in (1, 0):
                folder = root / f"producer-{worker}"
                folder.mkdir()
                atomic_json(
                    folder / "progress.json",
                    {
                        "chunks": [
                            {
                                "file": "0.gz",
                                "qwen_tokens": 20,
                                "records": 1,
                                "sha256": str(worker),
                            }
                        ]
                    },
                )
                manifest = publish_chunks(root, 2)
                if worker == 1:
                    prefix = manifest["chunks"][:]
            self.assertEqual(manifest["chunks"][: len(prefix)], prefix)
            self.assertEqual(manifest["qwen_tokens"], 40)
            self.assertEqual(len(available_chunks(root)), 2)
            self.assertEqual(publish_chunks(root, 10)["chunks"], manifest["chunks"])
            atomic_json(
                root / "producer-0" / "progress.json",
                {
                    "chunks": [
                        {
                            "file": "0.gz",
                            "qwen_tokens": 30,
                            "records": 1,
                            "sha256": "changed",
                        }
                    ]
                },
            )
            with self.assertRaises(ValueError):
                publish_chunks(root, 2)

    def test_archive_preserves_ids_and_pins_commit_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "original"
            output = root / "student"
            output.mkdir(parents=True)
            producer = root / "producer-0"
            producer.mkdir()
            chunk = producer / "0.gz"
            record = {
                "id": "example",
                "text": "sample",
                "ids": [4, 2],
                "qwen_ids": [8, 9, 151643],
            }
            with gzip.open(chunk, "wt") as handle:
                handle.write(json.dumps(record) + "\n")
            item = [
                0,
                0,
                {
                    "file": "0.gz",
                    "sha256": hashlib.sha256(chunk.read_bytes()).hexdigest(),
                },
            ]
            atomic_json(output / "dataset-plan.json", {"initial_chunks": [item]})
            atomic_json(output / "progress.json", {"tokens": 2, "target": 4500000000})

            class API:
                def whoami(self):
                    return {"name": "owner"}

                def create_repo(self, **kwargs):
                    assert kwargs["private"]

                def create_commit(self, **kwargs):
                    self.operations = kwargs["operations"]
                    return SimpleNamespace(oid="immutable-commit")

            api = API()
            with (
                patch("huggingface_hub.HfApi", return_value=api),
                patch(
                    "distillation.dataset_archive.time.monotonic",
                    side_effect=[0, 0, 90000],
                ),
            ):
                archive(
                    root, "student", Path(temp) / "fresh", lambda: None, lambda: None
                )
            result = json.loads((output / "dataset-archive.json").read_text())
            self.assertEqual(result["revision"], "immutable-commit")
            self.assertEqual(result["qwen_tokens"], 3)
            payload = api.operations[1].path_or_fileobj
            self.assertEqual(
                result["index_sha256"], hashlib.sha256(payload).hexdigest()
            )
            index = json.loads(payload)
            part = output / "dataset-archive" / "part-000000.jsonl.gz"
            self.assertEqual(
                index["parts"][0]["sha256"],
                hashlib.sha256(part.read_bytes()).hexdigest(),
            )
            with gzip.open(part, "rt") as handle:
                recovered = json.loads(handle.readline())
            self.assertEqual(recovered["qwen_ids"], record["qwen_ids"])
            self.assertEqual(recovered["text"], record["text"])
