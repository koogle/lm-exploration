"""Phase accounting must not add synchronizations or double-count nested samples."""

import unittest
from unittest.mock import MagicMock, patch

from trainer.timing import StepTiming


class StepTimingTests(unittest.TestCase):
    def test_disabled_has_no_cuda_work(self):
        with patch("trainer.timing.torch.cuda.Event") as event:
            timer = StepTiming()
            with timer.phase("forward"):
                pass
            self.assertEqual(timer.result(), {})
            event.assert_not_called()

    def test_repeated_and_nested_phases_are_reported_separately(self):
        events = [MagicMock() for _ in range(6)]
        for event, duration in zip(events[::2], [10, 3, 4], strict=True):
            event.elapsed_time.return_value = duration
        with patch("trainer.timing.torch.cuda.Event", side_effect=events), patch(
            "trainer.timing.torch.cuda.synchronize"
        ) as synchronize:
            timer = StepTiming(True)
            with timer.phase("accumulate"):
                with timer.phase("forward"):
                    pass
                with timer.phase("forward"):
                    pass
            totals = timer.result()
            self.assertEqual(totals["accumulate"]["cuda_stream_ms"], 10)
            self.assertEqual(totals["forward"]["cuda_stream_ms"], 7)
            for event in events:
                event.record.assert_called_once()
                event.synchronize.assert_not_called()
            synchronize.assert_not_called()
