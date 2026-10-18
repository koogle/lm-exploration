import time
import unittest
from typing import ClassVar

import torch
from training_data.prefetch import PrefetchedBatches


class SequenceData:
    identity: ClassVar = {"hash": "test"}
    sources: ClassVar = {"one": {"weight_percent": 100}}
    manifest: ClassVar = {}

    def __init__(self):
        self.n = 0

    def next_batch(self, batch_size, device="cpu"):
        if self.n >= 30:
            raise StopIteration
        time.sleep(0.003)
        self.n += 1
        return torch.full((batch_size, 2), self.n), torch.full((batch_size, 2), self.n)

    def state_dict(self):
        return {"n": self.n, "loss_token_exposures": {"one": self.n * 4}}

    def load_state_dict(self, state):
        self.n = state["n"]


class PrefetchTests(unittest.TestCase):
    def test_only_acknowledged_batches_survive_restart(self):
        data = PrefetchedBatches(SequenceData, 2, depth=3)
        for index in range(3):
            x, _ = data.next_batch(2)
            self.assertTrue((x == index + 1).all())
            data.acknowledge()
        time.sleep(0.03)  # Allow the producer to run ahead.
        self.assertLessEqual(data.queue.qsize(), 3)
        x, _ = data.next_batch(2)  # Delivered, not trained.
        saved = data.state_dict()
        self.assertEqual(saved["n"], 3)
        data.close()
        resumed = PrefetchedBatches(SequenceData, 2, depth=3, state=saved)
        x, _ = resumed.next_batch(2)
        self.assertTrue((x == 4).all())
        resumed.acknowledge()
        self.assertEqual(resumed.state_dict()["n"], 4)
        resumed.close()

    def test_producer_error_and_end_propagate(self):
        class Broken(SequenceData):
            def next_batch(self, *args):
                raise ValueError("reader failure")

        data = None
        with self.assertRaisesRegex(ValueError, "reader failure"):
            try:
                data = PrefetchedBatches(Broken, 2)
                data.next_batch(2)
            finally:
                if data:
                    data.close()
        data = PrefetchedBatches(SequenceData, 2, state={"n": 30})
        with self.assertRaises(StopIteration):
            data.next_batch(2)
        data.close()


if __name__ == "__main__":
    unittest.main()
