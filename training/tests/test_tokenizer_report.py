import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tokenizer_report import bootstrap, generate


class ReportTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_proportional_improvement(self):
        def measurement(scale):
            return {
                "metrics": {
                    "bytes": 300,
                    "parent_totals": [
                        {"source": "a", "parent_sha256": "p", "bytes": 100, "tokens": 50 * scale},
                        {"source": "a", "parent_sha256": "q", "bytes": 200, "tokens": 100 * scale},
                    ],
                }
            }

        old = bootstrap(measurement(1))
        new = bootstrap(measurement(0.8))
        np.testing.assert_allclose(100 * (1 - old / new), 20)

    def test_complete_wire_fixture_renders_comparisons(self):
        # Synthetic wire-contract fixtures stay in this temporary directory;
        # they are never inputs to the retained research report.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            sources = {
                f"source_{i}": {"bytes": 100, "tokens": 25, "bytes_per_token": 4} for i in range(12)
            }
            for size in (64, 128, 256, 512, 1024, 2048):
                for vocab in (16384, 32768, 50000):
                    result = {
                        "execution": "modal-cpu",
                        "size_mib": size,
                        "requested_vocab_size": vocab,
                        "actual_vocab_size": vocab,
                        "corpus_manifest_sha256": "synthetic-test-only",
                        "correctness_passed": True,
                        "requested_vocabulary_reached": True,
                        "training_seconds": 1,
                        "peak_process_rss_mib": 1,
                        "metrics": {
                            "bytes": 1200,
                            "tokens": 300,
                            "bytes_per_token": 4,
                            "sources": sources,
                            "vocabulary_use_fraction": 0.5,
                            "evaluation_mib_per_second": 1,
                            "fragmentation": {"example": 1},
                            "parent_totals": [
                                {"source": n, "parent_sha256": "parent", "bytes": 100, "tokens": 25}
                                for n in sources
                            ],
                        },
                    }
                    (path / f"{size}mib-{vocab}.json").write_text(json.dumps(result))
            results = generate(path)
            self.assertEqual(len(results), 18)
            report = (path / "REPORT.md").read_text()
            self.assertIn("**Status: complete.**", report)
            self.assertIn("final doubling", report)
            self.assertTrue((path / "source-comparisons.png").exists())
            self.assertTrue((path / "fragmentation-probes.png").exists())

    def test_mixed_tokenizer_rules_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for size in (64, 128):
                result = {
                    "execution": "modal-cpu",
                    "size_mib": size,
                    "requested_vocab_size": 16384,
                    "corpus_manifest_sha256": "same",
                    "config_sha256": str(size),
                }
                (path / f"{size}mib-16384.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                generate(path)

    def test_mixed_evaluation_corpora_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for size in (64, 128):
                result = {
                    "execution": "modal-cpu",
                    "size_mib": size,
                    "requested_vocab_size": 16384,
                    "corpus_manifest_sha256": "same",
                    "evaluation_manifest_sha256": str(size),
                }
                (path / f"{size}mib-16384.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "evaluation_manifest_sha256"):
                generate(path)

    def test_incomplete_report_does_not_invent_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            self.assertEqual(generate(path), [])
            report = (path / "REPORT.md").read_text()
            self.assertIn("0 of 18", report)
            self.assertIn("incomplete matrix", report)
            self.assertGreater((path / "research-report.pdf").stat().st_size, 1000)
            self.assertGreater((path / "learning-curve.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
