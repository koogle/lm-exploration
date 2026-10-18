import copy
import tempfile
import unittest

import torch
from model_100m.model import GPT, ModelConfig
from test_live_stream import fixture
from training_data.live_stream import LiveTokenMixture
from training_data.prefetch import PrefetchedBatches


class OverlapTrainingTests(unittest.TestCase):
    def test_same_updates_and_resume_with_unconsumed_future_batches(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            path, reader, _ = fixture(folder)

            def factory():
                return LiveTokenMixture(path, context=16, allow_unreviewed=True, reader=reader)

            def train(prefetch, steps, saved=None):
                torch.manual_seed(123)
                model = GPT(
                    ModelConfig(vocab_size=32768, block_size=16, n_layer=1, n_head=2, n_embd=16)
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                if saved:
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                if prefetch:
                    data = PrefetchedBatches(
                        factory, 2, depth=4, state=saved["data"] if saved else None
                    )
                else:
                    data = factory()
                for _ in range(steps):
                    x, y = data.next_batch(2)
                    loss = torch.nn.functional.cross_entropy(
                        model(x).flatten(0, 1), y.flatten(), ignore_index=-100
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if prefetch:
                        data.acknowledge()
                result = {
                    "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "data": data.state_dict(),
                }
                data.close()
                return result

            sync = train(False, 24)
            overlapped = train(True, 24)
            first = train(True, 12)
            resumed = train(True, 12, first)
            self.assertEqual(sync["data"], overlapped["data"])
            self.assertEqual(overlapped["data"], resumed["data"])
            for key, value in sync["model"].items():
                self.assertTrue(torch.equal(value, overlapped["model"][key]), key)
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
