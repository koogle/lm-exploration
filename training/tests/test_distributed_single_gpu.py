"""Local mathematical checks, not a CPU training entrypoint or CUDA qualification."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from model_100m.model import GPT, ModelConfig
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed.config import DistributedConfig
from trainer.optimizers import make_optimizer
from trainer.subword import backward_batch, evaluate_sources, validation_blocks


class SingleRankParityTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Local one-rank DDP check requires Gloo")
    def test_single_rank_matches_plain_model(self):
        # Exercise production accumulation with one-rank DDP and compare with a
        # plain model's full-batch loss/update. CUDA/NCCL/BF16 remain remote checks.
        with tempfile.TemporaryDirectory() as temporary:
            dist.init_process_group(
                "gloo",
                init_method=Path(temporary, "rendezvous").as_uri(),
                rank=0,
                world_size=1,
            )
            try:
                with torch.random.fork_rng():
                    torch.manual_seed(43)
                    model = GPT(
                        ModelConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32)
                    )
                    reference = copy.deepcopy(model)
                    ddp = DDP(model)
                    x = torch.randint(4, 64, (6, 16))
                    y = torch.randint(4, 64, (6, 16))
                    y[:2] = -100  # Fully masked microbatch must still work.
                    y[2:, 9:] = -100
                    count = int((y != -100).sum())
                    actual_loss = backward_batch(ddp, x, y, count, 2, 1, bf16=False)
                    expected_loss = F.cross_entropy(
                        reference(x).flatten(0, 1),
                        y.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    (expected_loss / count).backward()
                    torch.testing.assert_close(actual_loss, expected_loss.detach())
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4)
                    for current in (model, reference):
                        torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0, foreach=True)
                        torch.optim.AdamW(
                            current.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
                        ).step()
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            finally:
                dist.destroy_process_group()


class TrainingMathTests(unittest.TestCase):
    def test_optimizer_uses_nondefault_recipe_after_json_roundtrip(self):
        config = DistributedConfig(
            optimizer="adamw",
            adam_beta1=0.8,
            adam_beta2=0.99,
            peak_lr=0.002,
            weight_decay=0.02,
            gradient_clip_norm=0.25,
            validation_windows_per_source=3,
            log_every=7,
        )
        restored = DistributedConfig(**json.loads(json.dumps(config.as_dict())))
        self.assertEqual(restored, config)
        model = GPT(ModelConfig(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8))
        optimizer = make_optimizer(model, restored, fused=False)
        group = optimizer.adamw.param_groups[0]
        self.assertEqual(group["betas"], (0.8, 0.99))
        self.assertEqual(group["lr"], 0.002)
        self.assertEqual(group["weight_decay"], 0.02)
        model(torch.zeros(1, 2, dtype=torch.long)).sum().backward()
        optimizer.step()
        self.assertTrue(optimizer.adamw.state)

    def test_validation_samples_exact_count_and_fails_if_split_is_too_small(self):
        data = SimpleNamespace(sources={"source": {}}, blocks=lambda _: (i for i in range(5)))
        self.assertEqual(validation_blocks(data, 3), {"source": [0, 1, 2]})
        self.assertEqual(validation_blocks(data, 1), {"source": [0]})
        with self.assertRaisesRegex(ValueError, "fewer than 6"):
            validation_blocks(data, 6)

    def test_validation_weights_unequal_windows_by_valid_targets(self):
        with torch.random.fork_rng():
            torch.manual_seed(12)
            model = torch.nn.Embedding(8, 8)
            blocks = [
                {"input_ids": [1, 2, 3], "labels": [2, 3, 4], "loss_tokens": 3},
                {"input_ids": [5, 0, 0], "labels": [6, -100, -100], "loss_tokens": 1},
            ]
            stats = evaluate_sources(model, {"source": blocks}, device="cpu", bf16=False)
            loss = stats["per_source"]["source"]
            self.assertEqual(stats["per_stratum"], stats["per_source"])
            self.assertEqual(stats["target_counts"], {"source": 4})
            expected = F.cross_entropy(
                model(torch.tensor([1, 2, 3, 5])), torch.tensor([2, 3, 4, 6])
            )
            self.assertAlmostEqual(loss, expected.item(), places=6)
