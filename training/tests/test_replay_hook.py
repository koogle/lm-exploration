import tempfile
import unittest
from pathlib import Path

import torch
from model_10m.model import GPT, ModelConfig
from training_data.subword import load_frozen

from tests.test_replay import snapshot
from trainer.replay_hook import ReplayHook, ReplaySettings

ON = ReplaySettings(
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
        snapshot(Path(self.folder.name, "snapshot"))
        self.tokenizer = load_frozen()
        torch.manual_seed(0)
        self.model = GPT(
            ModelConfig(vocab_size=32768, block_size=16, n_layer=1, n_head=2, n_embd=16)
        )

    def tearDown(self):
        self.folder.cleanup()

    def hook(self, settings=ON):
        return ReplayHook.attach(settings, self.folder.name, self.tokenizer, 16, 0)

    def test_off_unless_configured(self):
        self.assertIsNone(self.hook(ReplaySettings()))
        with self.assertRaises(ValueError):
            ReplaySettings(replay_batch_size=2).check_replay()
        with self.assertRaises(ValueError):
            self.hook(ReplaySettings(**{**ON.__dict__, "replay_context": 32}))

    def test_loss_is_differentiable_and_validation_is_per_source(self):
        hook = self.hook()
        loss = hook.loss(self.model)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        validation = hook.validation(self.model)
        self.assertEqual(sorted(validation), ["code", "web"])
        self.assertEqual(validation, hook.validation(self.model))

    def test_state_restores_the_window_stream(self):
        hook = self.hook()
        state = hook.state()
        first = [hook.data.sample() for _ in range(3)]
        hook.load_state(state)
        self.assertEqual(first, [hook.data.sample() for _ in range(3)])


if __name__ == "__main__":
    unittest.main()
