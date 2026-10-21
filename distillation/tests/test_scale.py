import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from distillation.scale_data import atomic_json, available_chunks
from distillation.scale_train import Stream


class ScaleTests(unittest.TestCase):
    def test_publication_preserves_consumed_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for worker in range(4):
                folder = root / f"producer-{worker}"
                folder.mkdir()
                chunks = [
                    {"file": f"{i}.gz", "tokens": 100}
                    for i in range(2 if worker == 0 else 1)
                ]
                atomic_json(
                    folder / "progress.json", {"status": "running", "chunks": chunks}
                )
            initial = available_chunks(root)
            self.assertEqual(
                [(w, i) for w, i, _ in initial],
                [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)],
            )
            for worker in range(1, 4):
                atomic_json(
                    root / f"producer-{worker}" / "progress.json",
                    {
                        "status": "running",
                        "chunks": [
                            {"file": f"{i}.gz", "tokens": 100} for i in range(2)
                        ],
                    },
                )
            extended = available_chunks(root)
            self.assertEqual(extended[: len(initial)], initial)
            self.assertEqual(len(extended), 8)

    def test_stream_partial_cursor_and_completed_uneven_producers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for worker in range(4):
                folder = root / f"producer-{worker}"
                folder.mkdir()
                chunks = []
                for index in range(2 if worker == 0 else 1):
                    path = folder / f"chunk-{index}.jsonl.gz"
                    with gzip.open(path, "wt") as handle:
                        handle.write(
                            json.dumps(
                                {"ids": [4, 5, 6, 2], "qwen_ids": [42, 43, 151643]}
                            )
                            + "\n"
                        )
                    chunks.append(
                        {
                            "file": path.name,
                            "tokens": 4,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                atomic_json(
                    folder / "progress.json", {"status": "completed", "chunks": chunks}
                )
            self.assertEqual(len(available_chunks(root)), 5)
            stream = Stream(root)
            self.assertEqual(stream.next(2), ([1, 4], [4, 5]))
            resumed = Stream(root, stream.cursor)
            self.assertEqual(resumed.next(2), ([5, 6], [6, 2]))
            targets = [4, 5, 6, 2]
            while (item := resumed.next()) is not None:
                targets.extend(item[1])
            self.assertEqual(targets, [4, 5, 6, 2] * 5)
            qwen = Stream(root, large=True)
            self.assertEqual(qwen.next(2), ([151643, 42], [42, 43]))
            self.assertEqual(
                Stream(root, qwen.cursor, large=True).next(), ([43], [151643])
            )


if __name__ == "__main__":
    unittest.main()
