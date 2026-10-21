import json
import tempfile
import unittest
from pathlib import Path

from trainer.python_workflow import segment_attempt


class WorkerRestartTests(unittest.TestCase):
    def test_preemption_uses_only_complete_checkpoint_and_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, resume, report = segment_attempt(root, "run", 0)
            self.assertEqual((resume, report), ("", None))
            first.mkdir()
            (first / "step-000001000.pt").write_bytes(b"checkpoint")
            (first / "step-000002000.pt.partial").write_bytes(b"interrupted")
            (first / "latest.json").write_text(
                json.dumps({"step": 1000, "checkpoint": "step-000001000.pt"})
            )
            output, resume, report = segment_attempt(root, "run", 0)
            self.assertNotEqual(output, first)
            self.assertEqual(Path(resume).name, "step-000001000.pt")
            self.assertIsNone(report)
            self.assertTrue((first / "step-000002000.pt.partial").exists())

    def test_completed_input_retry_returns_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _, _ = segment_attempt(root, "run", 0)
            first.mkdir()
            checkpoint = first / "step-000000002.pt"
            checkpoint.write_bytes(b"checkpoint")
            expected = {"final_checkpoint": str(checkpoint), "steps": 2}
            (first / "report.json").write_text(json.dumps(expected))
            output, resume, report = segment_attempt(root, "run", 0)
            self.assertIsNone(output)
            self.assertEqual(report, expected)
            self.assertEqual(resume, str(checkpoint))
