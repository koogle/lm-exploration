import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training_data.overnight_snapshot import check_variety
from training_data.subword import TOKENIZER_SHA256, digest

from distributed_training.config import DistributedConfig
from distributed_training.run_validation import validate_overnight_run
from trainer.representative_validation import load_heldout


class OvernightGateTests(unittest.TestCase):
    def test_fresh_capacity_and_variety_are_both_required(self):
        db = sqlite3.connect(":memory:")
        db.execute(
            "CREATE TABLE eligible (stratum TEXT,tokens INTEGER,fresh INTEGER,source TEXT,split TEXT)"
        )
        db.executemany(
            "INSERT INTO eligible VALUES (?,?,?,?,?)",
            [
                (f"code/lang{i}/{task}", 400_000_000, 1, "code", "train")
                for i in range(8)
                for task in ("tests", "implementation")
            ],
        )
        with self.assertRaisesRegex(ValueError, "three path-task"):
            check_variety(db)
        db.execute("INSERT INTO eligible VALUES ('code/lang0/tooling',10000000,1,'code','train')")
        result = check_variety(db)
        self.assertEqual(len(result["languages"]), 8)
        # 6.11B passes a coarse 6.1B check but cannot satisfy the frozen
        # snapshot's 2% margin. Request more data before writing a database.
        db.execute("UPDATE eligible SET tokens=381250000 WHERE stratum NOT LIKE '%tooling'")
        with self.assertRaisesRegex(ValueError, "new eligible"):
            check_variety(db)
        db.execute("UPDATE eligible SET fresh=0")
        with self.assertRaisesRegex(ValueError, "new eligible"):
            check_variety(db)
        db.close()

    def test_heldout_integrity_coverage_and_old_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sample = folder / "heldout.gz"
            with gzip.open(sample, "wt") as out:
                for name in ("code", "eu"):
                    out.write(json.dumps({"source": name, "tokens": [1, 4, 5, 2]}) + "\n")
            path = folder / "manifest.json"
            manifest = {
                "sources": {"code": {}, "eu": {}},
                "heldout": {
                    "path": sample.name,
                    "sha256": digest(sample),
                    "tokenizer_sha256": TOKENIZER_SHA256,
                    "windows": 2,
                },
            }
            path.write_text(json.dumps(manifest))
            rows = load_heldout(path, None)
            self.assertEqual(
                rows[0],
                (
                    "code",
                    {
                        "input_ids": [1, 4, 5],
                        "labels": [4, 5, 2],
                        "loss_tokens": 3,
                        "stratum": "code",
                    },
                ),
            )
            sample.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_heldout(path, None)

    def test_ten_billion_limit_belongs_to_overnight_run(self):
        settings = {
            "world_size": 8,
            "microbatch": 2,
            "target_tokens": 10_000_000_000,
            "allow_unreviewed_data": True,
        }
        validate_overnight_run(DistributedConfig(**settings))
        for change in (
            {"target_tokens": 10_000_000_001},
            {"n_layer": 8, "n_head": 3, "n_embd": 192},
            {"world_size": 3},
            {"pilot": True},
        ):
            # General settings are valid; only this experiment rejects them.
            config = DistributedConfig(**{**settings, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_overnight_run(config)
