"""Local mathematical checks, not a CPU training entrypoint or CUDA qualification."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from model_100m.model import GPT, ModelConfig
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.loop import backward_batch
from workflows.config import load_workflow, resolve_payload


class SingleRankParityTests(unittest.TestCase):
    def test_pilot_recipes_only_change_world_size(self):
        root = Path(__file__).resolve().parents[1] / "workflows"
        single = resolve_payload(load_workflow(root / "h100x1-pilot.json"))[2]
        multi = resolve_payload(load_workflow(root / "h100x2-pilot.json"))[2]
        self.assertEqual(single.world_size, 1)
        self.assertEqual(multi.world_size, 2)
        settings = multi.as_dict()
        settings["world_size"] = 1
        self.assertEqual(single.as_dict(), settings)
        self.assertEqual(single.global_windows // single.microbatch, 2)

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


if __name__ == "__main__":
    unittest.main()
