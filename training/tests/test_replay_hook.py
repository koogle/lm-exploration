import tempfile
import unittest
from pathlib import Path

import torch
from model_10m.model import GPT, ModelConfig
from training_data.raw_token_stream import RawTokenMixture

from tests.test_source_budgets import snapshot
from trainer.replay_hook import ReplayHook, ReplaySettings, summed_loss

ON = ReplaySettings(
    replay=True,
    replay_snapshot="snapshot",
    replay_sources={"code": 1, "web": 1},
    replay_batch_size=2,
    replay_context=4,
    replay_weight=0.2,
    replay_validation_windows=1,
)


class ReplayHookTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        Path(self.folder.name, "snapshot").mkdir()
        self.manifest, _ = snapshot(
            Path(self.folder.name, "snapshot"), {"code": {}, "web": {}, "books": {}}
        )
        torch.manual_seed(0)
        self.model = GPT(
            ModelConfig(vocab_size=32768, block_size=16, n_layer=1, n_head=2, n_embd=16)
        )

    def tearDown(self):
        self.folder.cleanup()

    def hook(self, settings=ON):
        return ReplayHook.attach(settings, self.folder.name, 16)

    def test_off_unless_switched_on_and_complete(self):
        self.assertIsNone(self.hook(ReplaySettings()))
        self.assertIsNone(self.hook(ReplaySettings(**{**ON.__dict__, "replay": False})))
        with self.assertRaises(ValueError):
            ReplaySettings(replay=True).check_replay()
        with self.assertRaises(ValueError):
            ReplaySettings(**{**ON.__dict__, "replay_weight": 0.0}).check_replay()
        with self.assertRaises(ValueError):
            self.hook(ReplaySettings(**{**ON.__dict__, "replay_context": 32}))

    def test_reads_only_the_weighted_sources(self):
        mixture = RawTokenMixture(self.manifest, 4, weights={"web": 1})
        self.assertEqual({mixture.next_block()["source"] for _ in range(10)}, {"web"})
        with self.assertRaises(ValueError):
            RawTokenMixture(self.manifest, 4, weights={"news": 1})

    def test_backward_adds_weighted_replay_gradients(self):
        hook = self.hook()
        state = hook.state()
        loss = hook.backward(self.model)
        added = self.model.token_embedding.weight.grad.clone()
        self.model.zero_grad()
        hook.load_state(state)
        value, count = summed_loss(self.model, *hook.data.next_batch(ON.replay_batch_size))
        (value / count).backward()
        self.assertAlmostEqual(loss, (value / count).item(), places=5)
        torch.testing.assert_close(added, ON.replay_weight * self.model.token_embedding.weight.grad)

    def test_validation_is_per_source_and_fixed(self):
        hook = self.hook()
        validation = hook.validation(self.model)
        self.assertEqual(sorted(validation), ["code", "web"])
        self.assertEqual(validation, self.hook().validation(self.model))

    def test_state_restores_the_window_stream(self):
        hook = self.hook()
        state = hook.state()
        first = [hook.data.next_batch(2) for _ in range(3)]
        hook.load_state(state)
        for x, y in first:
            again = hook.data.next_batch(2)
            self.assertTrue(torch.equal(x, again[0]) and torch.equal(y, again[1]))


if __name__ == "__main__":
    unittest.main()
