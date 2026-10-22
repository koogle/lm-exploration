import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from audit_targets import audit


class TargetArchiveTests(unittest.TestCase):
    def check(self, selected=None, tail=0.25):
        logp = math.log(0.75 / 64)
        row = {
            "id": "fixture:0",
            "token_ids": [0],
            "accepted": True,
            "finish_reason": "stop",
            "targets": {
                "topk_ids": [list(range(64))],
                "topk_logprobs": [[logp] * 64],
                "selected_logprobs": [logp if selected is None else selected],
                "tail_mass": [tail],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            return audit(path)

    def test_round_trip_preserves_full_mass(self):
        self.assertTrue(self.check()["passed"])

    def test_corrupted_selected_probability_and_tail_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.check(selected=-10)
        with self.assertRaisesRegex(ValueError, "mass"):
            self.check(tail=0)
