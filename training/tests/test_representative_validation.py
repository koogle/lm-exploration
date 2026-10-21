import gzip
import json
import tempfile
import unittest
from pathlib import Path

from training_data.subword import TOKENIZER_SHA256, digest

from trainer.representative_validation import load_heldout


class RepresentativeValidationTests(unittest.TestCase):
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
