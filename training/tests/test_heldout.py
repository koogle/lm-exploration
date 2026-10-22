import json
import tempfile
import unittest
from pathlib import Path

import torch
from training_data.heldout import load_heldout, write_heldout
from training_data.subword import TOKENIZER_SHA256

from distributed.config import DistributedConfig, DistributedRuntime
from trainer.loop import evaluate_sources, train


def snapshot(folder, rows, sources=("code", "web")):
    entry = write_heldout(folder, rows)
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps({"sources": {name: {} for name in sources}, "heldout": entry}))
    return path, entry


ROWS = (
    {"source": "code", "stratum": "code/python", "tokens": [1, 4, 5, 2], "parent": "a/b"},
    {"source": "code", "stratum": "code/rust", "tokens": [1, 6, 7]},
    {"source": "web", "tokens": [1, 8, 9, 10, 2]},
)


class HeldoutTests(unittest.TestCase):
    def test_round_trip_groups_by_source_with_strata(self):
        with tempfile.TemporaryDirectory() as folder:
            path, entry = snapshot(folder, ROWS)
            self.assertEqual(entry["windows"], 3)
            self.assertEqual(entry["tokenizer_sha256"], TOKENIZER_SHA256)
            heldout = load_heldout(path, context=4)
            self.assertEqual(
                heldout["code"][0],
                {
                    "input_ids": [1, 4, 5],
                    "labels": [4, 5, 2],
                    "loss_tokens": 3,
                    "stratum": "code/python",
                },
            )
            self.assertEqual(heldout["web"][0]["stratum"], "web")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_heldout(folder, ROWS)

    def test_rejects_corruption_coverage_and_invalid_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = snapshot(folder, ROWS)
            with self.assertRaisesRegex(ValueError, "Invalid held-out window"):
                load_heldout(path, context=3)
            (Path(folder) / "heldout.jsonl.gz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_heldout(path, context=4)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = snapshot(folder, ROWS[:2])
            with self.assertRaisesRegex(ValueError, "cover every source"):
                load_heldout(path, context=4)
        with tempfile.TemporaryDirectory() as folder:
            bad = [*ROWS, {"source": "web", "tokens": [1, 40000]}]
            path, _ = snapshot(folder, bad)
            with self.assertRaisesRegex(ValueError, "Invalid held-out token"):
                load_heldout(path, context=4)

    def test_per_stratum_losses_weight_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = snapshot(folder, ROWS)
            heldout = load_heldout(path, context=4)
        torch.manual_seed(3)
        model = torch.nn.Embedding(16, 16)
        stats = evaluate_sources(model, heldout, device="cpu", bf16=False)
        self.assertEqual(set(stats["per_stratum"]), {"code/python", "code/rust", "web"})
        self.assertEqual(
            stats["stratum_target_counts"], {"code/python": 3, "code/rust": 2, "web": 4}
        )
        self.assertEqual(stats["target_counts"], {"code": 5, "web": 4})
        self.assertEqual(stats["windows"], 3)
        code = (stats["per_stratum"]["code/python"] * 3 + stats["per_stratum"]["code/rust"] * 2) / 5
        self.assertAlmostEqual(stats["per_source"]["code"], code, places=6)

    def test_stop_tokens_must_lie_within_the_target(self):
        config = DistributedConfig(
            world_size=1, global_windows=16, target_tokens=100, warmup_tokens=1
        )
        runtime = DistributedRuntime(0, 0, 1, "127.0.0.1", 29500)
        for value in (0, 101):
            with self.subTest(value), self.assertRaisesRegex(ValueError, "stop_tokens"):
                train(config, "manifest.json", "out", runtime=runtime, stop_tokens=value)


if __name__ == "__main__":
    unittest.main()
