import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from distillation.scale_data import atomic_json, available_chunks
from distillation.scale_train import Stream, rank_batches


class ScaleTests(unittest.TestCase):
    def test_rank_partition_and_padding_preserve_global_token_gradient(self):
        import torch
        from torch.nn import functional as F

        torch.manual_seed(7)
        windows = [([0, 1, 2], [1, 2, 0]), ([1], [2]), ([2, 0], [0, 1])]
        reference = torch.nn.Embedding(3, 3)
        total = sum(len(y) for _, y in windows)
        loss = (
            sum(
                F.cross_entropy(
                    reference(torch.tensor(x)), torch.tensor(y), reduction="sum"
                )
                for x, y in windows
            )
            / total
        )
        loss.backward()
        gradients = []
        seen = []
        for rank in range(8):
            model = torch.nn.Embedding(3, 3)
            model.load_state_dict(reference.state_dict())
            batches = rank_batches(windows, rank, 8, 1, 0)
            self.assertEqual(len(batches), 4)
            for batch in batches:
                for x, y in batch:
                    seen.extend(t for t in y if t != -100)
                    (
                        F.cross_entropy(
                            model(torch.tensor(x)),
                            torch.tensor(y),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        * 8
                        / total
                    ).backward()
            gradients.append(model.weight.grad)
        torch.testing.assert_close(
            torch.stack(gradients).mean(0), reference.weight.grad
        )
        self.assertEqual(seen, [1, 2, 0, 2, 0, 1])

    def test_fresh_continuation_is_append_only_and_never_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            for root, value in ((first, 10), (second, 20)):
                for worker in range(4):
                    folder = root / f"producer-{worker}"
                    folder.mkdir(parents=True)
                    path = folder / "chunk.jsonl.gz"
                    with gzip.open(path, "wt") as handle:
                        handle.write(
                            json.dumps({"qwen_ids": [value + worker, 151643]}) + "\n"
                        )
                    atomic_json(
                        folder / "progress.json",
                        {
                            "status": "running" if root == first else "completed",
                            "chunks": [
                                {
                                    "file": path.name,
                                    "tokens": 2,
                                    "sha256": hashlib.sha256(
                                        path.read_bytes()
                                    ).hexdigest(),
                                }
                            ],
                        },
                    )
            stream = Stream(first, large=True, continuation=second)
            targets = []
            while (item := stream.next()) is not None:
                targets.extend(item[1])
            self.assertEqual(targets, [10, 151643, 11, 151643, 12, 151643, 13, 151643])
            self.assertEqual(stream.cursor["chunk"], 4)
            # Freeze the original prefix while its producer jobs are still running.
            immediate = Stream(
                first,
                stream.cursor,
                large=True,
                continuation=second,
                initial_chunks=available_chunks(first),
            )
            self.assertEqual(immediate.next()[1], [20, 151643])

            for worker in range(4):
                path = first / f"producer-{worker}" / "progress.json"
                state = json.loads(path.read_text())
                state["status"] = "completed"
                atomic_json(path, state)
            resumed = Stream(first, stream.cursor, large=True, continuation=second)
            self.assertEqual(resumed.next(1), ([151643], [20]))
            resumed_again = Stream(
                first, resumed.cursor, large=True, continuation=second
            )
            targets = []
            while (item := resumed_again.next()) is not None:
                targets.extend(item[1])
            self.assertEqual(targets, [151643, 21, 151643, 22, 151643, 23, 151643])
            self.assertIsNone(resumed_again.next())
            self.assertIsNone(
                Stream(
                    first, resumed_again.cursor, large=True, continuation=second
                ).next()
            )

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
            repeating = Stream(root, large=True, repeat=True)
            for _ in range(5):
                self.assertEqual(repeating.next()[1], [42, 43, 151643])
            resumed_repeat = Stream(root, repeating.cursor, large=True, repeat=True)
            self.assertEqual(resumed_repeat.next(1), ([151643], [42]))
            self.assertEqual(resumed_repeat.cursor["pass"], 1)
            self.assertEqual(
                Stream(root, resumed_repeat.cursor, large=True, repeat=True).next()[1],
                [43, 151643],
            )
            state_path = root / "producer-3" / "progress.json"
            state = json.loads(state_path.read_text())
            state["status"] = "running"
            atomic_json(state_path, state)
            waiting = Stream(root, large=True, repeat=True)
            while waiting.next() is not None:
                pass
            self.assertEqual(waiting.cursor["pass"], 0)
            qwen = Stream(root, large=True)
            self.assertEqual(qwen.next(2), ([151643, 42], [42, 43]))
            self.assertEqual(
                Stream(root, qwen.cursor, large=True).next(), ([43], [151643])
            )


if __name__ == "__main__":
    unittest.main()
