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
