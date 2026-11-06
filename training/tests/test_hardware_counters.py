"""Counter access requires recorded metrics, not just a successful CUDA launch."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_hybrid.hardware_counters import measured_rows, probe


class CounterProbeTests(unittest.TestCase):
    def test_denied_is_not_measurement(self):
        self.assertEqual(measured_rows("==ERROR== ERR_NVGPUCTRPERM"), [])

    def test_csv_preamble(self):
        rows = measured_rows(
            '==PROF== Connected\n"ID","Metric Name","Metric Unit","Metric Value"\n'
            '"0","sm__cycles_elapsed.avg","cycle","123"\n'
        )
        self.assertEqual(rows[0]["Metric Value"], "123")

    def test_zero_exit_without_metrics_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("model_hybrid.hardware_counters.executable", return_value="ncu"),
                patch("model_hybrid.hardware_counters.subprocess.run") as run,
                patch(
                    "model_hybrid.hardware_counters.subprocess.check_output",
                    return_value="2025.3.1",
                ),
            ):
                run.return_value.returncode = 0
                run.return_value.stdout = "==WARNING== No kernels profiled"
                run.return_value.stderr = ""
                result = probe(folder)
            self.assertFalse(result["accessible"])
            self.assertTrue((Path(folder) / "probe.json").exists())


if __name__ == "__main__":
    unittest.main()
