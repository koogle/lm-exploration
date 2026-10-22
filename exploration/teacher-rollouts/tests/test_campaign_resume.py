import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from campaign_resume import resume_state


class ResumeTests(unittest.TestCase):
    def test_retry_adopts_calls_without_extending_deadline(self):
        old = {
            "backend": "vllm",
            "batch_size": 32,
            "workers": 1,
            "status": "generating",
            "resume_epoch": "resume1",
            "deadline": 200,
            "calls": {"0": "fc-live"},
        }
        config = {**old, "resume_deadline": 999, "resume_requested_at": 100}
        self.assertEqual(resume_state(old, config, 150), old)

    def test_explicit_resume_preserves_history_and_changes_window_once(self):
        old = {
            "backend": "vllm",
            "batch_size": 32,
            "workers": 1,
            "status": "generation_window_completed",
            "deadline": 50,
            "calls": {},
        }
        config = {
            **old,
            "resume_epoch": "resume1",
            "resume_deadline": 200,
            "resume_requested_at": 100,
        }
        result = resume_state(old, config, 110)
        self.assertEqual(result["deadline"], 200)
        self.assertEqual(result["history"][0]["deadline"], 50)
        with self.assertRaises(ValueError):
            resume_state({**old, "status": "generating"}, config, 110)

    def invoke(self, now, interrupted=False):
        import campaign_modal as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "test"
            folder.mkdir()
            state = {
                "backend": "vllm",
                "batch_size": 32,
                "workers": 1,
                "status": "generating",
                "deadline": 200,
                "calls": {"0": "fc-live"},
                "archive_call": "fc-archive",
                "correctness_passed": True,
                "throughput_compared": True,
            }
            (folder / "campaign.json").write_text(json.dumps(state))
            call = Mock()
            if interrupted:
                call.get.side_effect = KeyboardInterrupt(
                    "simulated container interruption"
                )
            with (
                patch.object(
                    module,
                    "Path",
                    side_effect=lambda p: root if str(p) == "/runs" else Path(p),
                ),
                patch.object(module, "volume"),
                patch.object(module.time, "time", return_value=now),
                patch.object(module.modal.FunctionCall, "from_id", return_value=call),
            ):
                if interrupted:
                    with self.assertRaises(KeyboardInterrupt):
                        module.supervise.local("test", state)
                else:
                    module.supervise.local("test", state)
            return call, json.loads((folder / "campaign.json").read_text())

    def test_interrupted_controller_keeps_teachers_and_saved_handles(self):
        call, state = self.invoke(100, interrupted=True)
        call.cancel.assert_not_called()
        self.assertEqual(state["calls"], {"0": "fc-live"})
        self.assertEqual(state["status"], "generating")

    def test_deadline_still_cancels_workers(self):
        call, state = self.invoke(201)
        call.cancel.assert_called_once_with(terminate_containers=True)
        self.assertEqual(state["status"], "generation_window_completed")
