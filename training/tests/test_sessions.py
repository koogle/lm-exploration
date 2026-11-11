import json
import tempfile
import unittest
from pathlib import Path

from workflows.sessions import continued_checkpoint, newest_checkpoint, session_state


def session(root, name, steps=(), report=None, latest=True):
    folder = Path(root) / name
    folder.mkdir()
    for step in steps:
        (folder / f"step-{step:09d}.pt").write_bytes(b"weights")
    if steps and latest:
        (folder / "latest.json").write_text(json.dumps({"checkpoint": f"step-{steps[-1]:09d}.pt"}))
    if report is not None:
        (folder / "report.json").write_text(json.dumps(report))
    return folder


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_session_state(self):
        self.assertEqual(session_state(self.root / "run-s00"), ("fresh", None))
        session(self.root, "run-s01")
        self.assertEqual(session_state(self.root / "run-s01"), ("interrupted", None))
        cut = session(self.root, "run-s02", steps=(10, 20))
        self.assertEqual(session_state(cut), ("interrupted", cut / "step-000000020.pt"))
        done = session(self.root, "run-s03", steps=(30,), report={"tokens": 3})
        self.assertEqual(session_state(done), ("finished", {"tokens": 3}))

    def test_latest_must_name_a_file_on_disk(self):
        folder = session(self.root, "run-s00", steps=(10,))
        (folder / "latest.json").write_text(json.dumps({"checkpoint": "step-000000099.pt"}))
        self.assertIsNone(newest_checkpoint(folder))
        (folder / "step-000000099.pt").write_bytes(b"")
        self.assertIsNone(newest_checkpoint(folder))

    def test_relaunch_continues_past_the_last_reported_session(self):
        session(self.root, "run-a3-s01", steps=(100,), report={})
        resume = "run-a3-s01/step-000000100.pt"
        self.assertEqual(continued_checkpoint(self.root, resume), resume)
        session(self.root, "run-a3-s02", steps=(150, 200))
        self.assertEqual(continued_checkpoint(self.root, resume), "run-a3-s02/step-000000200.pt")
        # A session cut off before its first checkpoint adds nothing.
        session(self.root, "run-a3-s03")
        self.assertEqual(continued_checkpoint(self.root, resume), "run-a3-s02/step-000000200.pt")
        # Another run's sessions and an empty resume are left alone.
        session(self.root, "run-s02", steps=(999,))
        self.assertEqual(continued_checkpoint(self.root, ""), "")
        self.assertEqual(
            continued_checkpoint(self.root, "other/step-000000001.pt"), "other/step-000000001.pt"
        )


if __name__ == "__main__":
    unittest.main()
