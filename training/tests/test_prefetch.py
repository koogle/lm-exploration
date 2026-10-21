import unittest
from typing import ClassVar

import torch
from training_data.prefetch import PrefetchedBatches


class CountingData:
    identity: ClassVar = {"hash": "test"}
    sources: ClassVar = {"one": {"weight_percent": 100}}
    manifest: ClassVar = {}

    def __init__(self):
        self.n = 0

    def next_batch(self, batch_size, device="cpu"):
        self.n += 1
        return torch.full((batch_size, 2), self.n), torch.full((batch_size, 2), self.n)

    def state_dict(self):
        return {"n": self.n, "loss_token_exposures": {"one": self.n * 4}}

    def load_state_dict(self, state):
        self.n = state["n"]


class PrefetchTests(unittest.TestCase):
    def test_restart_resumes_after_last_acknowledged_batch(self):
        data = PrefetchedBatches(CountingData, 2, depth=3)
        for _ in range(2):
            data.next_batch(2)
            data.acknowledge()
        data.next_batch(2)  # Delivered but not trained.
        saved = data.state_dict()
        data.close()
        resumed = PrefetchedBatches(CountingData, 2, depth=3, state=saved)
        x, _ = resumed.next_batch(2)
        resumed.close()
        self.assertTrue((x == 3).all())


if __name__ == "__main__":
    unittest.main()
