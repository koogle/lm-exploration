"""Eight-rank Muon step agreement and exact optimizer restore on CUDA."""

import copy
import os

import torch
import torch.distributed as dist
from model_100m.model import GPT, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed.config import DistributedConfig
from trainer.loop import backward_batch
from trainer.optimizers import make_optimizer

rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
try:
    torch.manual_seed(42)
    model = GPT(
        ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, eos_id=2)
    ).cuda()
    for block in model.blocks:
        block.attention.cache_rotary(32)
    model.compile(dynamic=True)
    ddp = DDP(model, device_ids=[rank], broadcast_buffers=False)
    config = DistributedConfig(optimizer="muon")
    optimizer = make_optimizer(model, config)
    x = torch.randint(0, 64, (4, 32), device="cuda")
    y = x.clone()
    y[:, 16:] = -100
    if rank == 0:
        y[:] = -100
    count = (y != -100).sum()
    dist.all_reduce(count)
    backward_batch(ddp, x, y, count, 2, world, crop_padding=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    for p in model.parameters():
        reference = p.detach().clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(p, reference, atol=0, rtol=0)
    restored = GPT(model.config).cuda()
    restored.load_state_dict(model.state_dict())
    resumed = make_optimizer(restored, config)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for a, b in zip(model.parameters(), restored.parameters()):
        b.grad = a.grad.detach().clone()
    optimizer.step()
    resumed.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if rank == 0:
        print("Muon eight-rank compiled BF16 step and restore: passed", flush=True)
finally:
    dist.destroy_process_group()
