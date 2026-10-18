"""Loss, gradient, masking and checkpoint-schema checks for execution optimizations."""

import copy
import unittest
from contextlib import nullcontext

import torch
from model_100m.model import GPT, ModelConfig

from trainer.distributed_train import backward_batch, crop_trailing_padding


class LocalDDP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def no_sync(self):
        return nullcontext()


class OptimizationTests(unittest.TestCase):
    def test_equivalence(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        y[0] = -100
        y[1, 3:] = -100
        y[2, 8:] = -100
        y[3, 12:] = -100
        y[3, 2:5] = -100
        count = int((y != -100).sum())
        reference = copy.deepcopy(model)
        expected = backward_batch(LocalDDP(reference), x, y, count, 1, 1, bf16=False)
        for microbatch in (1, 2, 4):
            for chunk in (0, 7):
                with self.subTest(microbatch=microbatch, chunk=chunk):
                    current = copy.deepcopy(model)
                    for block in current.blocks:
                        block.attention.cache_rotary(16)
                    self.assertEqual(set(current.state_dict()), set(model.state_dict()))
                    actual = backward_batch(
                        LocalDDP(current),
                        x,
                        y,
                        count,
                        microbatch,
                        1,
                        bf16=False,
                        crop_padding=True,
                        loss_chunk_size=chunk,
                    )
                    torch.testing.assert_close(actual, expected)
                    for a, b in zip(current.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)
                    restored = copy.deepcopy(model)
                    restored.load_state_dict(current.state_dict(), strict=True)
                    torch.testing.assert_close(restored(x), current(x))

    def test_optimizer_resume(self):
        torch.manual_seed(7)
        config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
        model = GPT(config)
        for block in model.blocks:
            block.attention.cache_rotary(8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randint(0, 32, (4, 8))
        y = x.clone()
        y[:, 5:] = -100

        def update(current, optim):
            optim.zero_grad(set_to_none=True)
            backward_batch(
                LocalDDP(current), x, y, 20, 2, 1, bf16=False, crop_padding=True, loss_chunk_size=3
            )
            optim.step()

        update(model, optimizer)
        payload = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        restored = GPT(config)
        restored.load_state_dict(payload["model"], strict=True)
        for block in restored.blocks:
            block.attention.cache_rotary(8)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
        restored_optimizer.load_state_dict(payload["optimizer"])
        update(model, optimizer)
        update(restored, restored_optimizer)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(optimizer.state.values(), restored_optimizer.state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_empty_and_interior_masks(self):
        x = torch.ones(2, 9, dtype=torch.long)
        y = torch.full_like(x, -100)
        self.assertEqual(crop_trailing_padding(x, y)[0].shape, (2, 1))
        y[1, 7] = 3
        self.assertEqual(crop_trailing_padding(x, y)[0].shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
