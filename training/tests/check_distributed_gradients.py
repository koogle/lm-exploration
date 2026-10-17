"""Reference test for the production accumulation path, on CUDA/NCCL."""

import copy
import os

import torch
import torch.distributed as dist
from model_100m.model import GPT, ModelConfig
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.distributed_train import backward_batch

rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
try:
    torch.manual_seed(43)
    model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32)).cuda()
    reference = copy.deepcopy(model)
    ddp = DDP(model, device_ids=[rank])
    generator = torch.Generator().manual_seed(9)
    x = torch.randint(4, 64, (12, 16), generator=generator).cuda()
    y = torch.randint(4, 64, (12, 16), generator=generator).cuda()
    for i in range(12):
        y[i, 3 + i :] = -100
    # Rank 0 can have entirely masked microbatches; they still join all collectives.
    y[0:2] = -100
    count = int((y != -100).sum())
    backward_batch(ddp, x.chunk(world)[rank], y.chunk(world)[rank], count, 2, world, bf16=False)
    loss = (
        F.cross_entropy(reference(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
        / count
    )
    loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4)
    for current in (model, reference):
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0, foreach=True)
        torch.optim.AdamW(
            current.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        ).step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
    if rank == 0:
        print(f"Production accumulation gradients and optimizer update match on {world} GPUs")
finally:
    dist.destroy_process_group()
