"""The 24-hour launch deadline bounds every call a workflow starts."""

import tempfile
import time
import unittest

from workflows.deadline import (
    MAX_LAUNCH_SECONDS,
    SHUTDOWN_SECONDS,
    bounded_timeout,
    cancel_at_deadline,
    new_deadline,
    recorded_call,
    segment_seconds,
    session_seconds,
    track,
    tracked_calls,
)


class DeadlineTests(unittest.TestCase):
    def test_deadline_is_inside_24_hours(self):
        self.assertLess(new_deadline(now=0), MAX_LAUNCH_SECONDS)

    def test_child_timeouts_end_before_the_deadline(self):
        deadline = new_deadline(now=0)
        self.assertEqual(bounded_timeout(deadline, 3600, now=0), 3600)
        self.assertEqual(bounded_timeout(deadline, 86400, now=80000), int(deadline - 80000))
        with self.assertRaisesRegex(RuntimeError, "refusing new work"):
            bounded_timeout(deadline, 3600, now=deadline - 60)

    def test_segment_leaves_time_to_checkpoint(self):
        self.assertEqual(segment_seconds(25200, 21600), 21600)
        self.assertEqual(segment_seconds(5000, 21600), 5000 - 480)
        with self.assertRaises(RuntimeError):
            segment_seconds(400, 21600)

    def test_session_queued_past_its_spawn_stops_before_the_deadline(self):
        deadline = 100_000.0
        # Spawned with 6 hours left, but its container starts 31 minutes later.
        start = deadline - 6 * 3600 + 31 * 60
        seconds = session_seconds(deadline, 6 * 3600 - SHUTDOWN_SECONDS, now=start)
        self.assertEqual(start + seconds, deadline - SHUTDOWN_SECONDS)
        self.assertEqual(session_seconds(deadline, 60, now=start), 60)
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            session_seconds(deadline, 3600, now=deadline - SHUTDOWN_SECONDS)

    def test_restarted_controller_reattaches_recorded_session(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as folder:
            child = SimpleNamespace(object_id="fc-child")
            spawn, reconnect, commit = Mock(return_value=child), Mock(return_value=child), Mock()
            first = recorded_call(folder, "train_segment:run-s00", spawn, reconnect, commit)
            second = recorded_call(folder, "train_segment:run-s00", spawn, reconnect, commit)
            self.assertIs(first, second)
            spawn.assert_called_once()
            reconnect.assert_called_once_with("fc-child")
            commit.assert_called_once()
            self.assertEqual(tracked_calls(folder), ["fc-child"])

    def test_duplicate_child_records_fail_without_spawning(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as folder:
            track(folder, "fc-a", "train_segment:run-s00")
            track(folder, "fc-b", "train_segment:run-s00")
            spawn = Mock()
            with self.assertRaisesRegex(RuntimeError, "Multiple recorded calls"):
                recorded_call(folder, "train_segment:run-s00", spawn, Mock(), Mock())
            spawn.assert_not_called()

    def test_calls_are_recorded_for_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(tracked_calls(folder), [])
            track(folder, "fc-b", "train_segment")
            track(folder, "fc-a", "continue_training")
            self.assertEqual(tracked_calls(folder), ["fc-a", "fc-b"])
            self.assertEqual(tracked_calls(folder, kind="continue_training"), ["fc-a"])


class SupervisorTests(unittest.TestCase):
    def test_cancels_recorded_calls_at_the_deadline(self):
        cancelled = []
        result = cancel_at_deadline(
            time.time() + 0.05, lambda: ["fc-a", "fc-b"], poll_seconds=0.01, cancel=cancelled.append
        )
        self.assertEqual((result, cancelled), (["fc-a", "fc-b"], ["fc-a", "fc-b"]))

    def test_finished_work_ends_the_wait_without_cancelling(self):
        cancelled = []
        result = cancel_at_deadline(
            time.time() + 3600, lambda: ["fc-a"], lambda: True, 0.01, cancelled.append
        )
        self.assertEqual((result, cancelled), (None, []))


if __name__ == "__main__":
    unittest.main()
