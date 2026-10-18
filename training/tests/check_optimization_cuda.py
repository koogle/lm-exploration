"""One/eight-rank correctness gate, including empty ranks and BF16 production execution."""

import copy
import json
import os

import torch
import torch.distributed as dist
from model_100m.model import GPT, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.distributed_train import backward_batch

rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
torch.set_num_threads(2)
dist.init_process_group("nccl")
try:
    torch.manual_seed(43)
    initial = GPT(ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)).cuda()
    x = torch.randint(4, 64, (32, 32), device="cuda")
    y = torch.randint(4, 64, (32, 32), device="cuda")
    for i in range(32):
        y[i, 3 + i % 24 :] = -100
    y[:4] = -100
    y[8, 2:5] = -100
    count = int((y != -100).sum())
    reference = copy.deepcopy(initial)
    expected = reference(x, labels=y) / count
    expected.backward()
    results = []
    for bf16 in (False, True):
        for microbatch, chunk, compiled in (
            (1, 0, False),
            (2, 0, False),
            (4, 0, False),
            (2, 7, False),
            (2, 0, True),
        ):
            model = copy.deepcopy(initial)
            for block in model.blocks:
                block.attention.cache_rotary(32)
            if compiled:
                model.compile(dynamic=True)
            ddp = DDP(model, device_ids=[rank], broadcast_buffers=False)
            loss = backward_batch(
                ddp,
                x.chunk(world)[rank],
                y.chunk(world)[rank],
                count,
                microbatch,
                world,
                bf16=bf16,
                crop_padding=True,
                loss_chunk_size=chunk,
            )
            dist.all_reduce(loss)
            torch.testing.assert_close(
                loss / count, expected, atol=0.003 if bf16 else 2e-6, rtol=0.003 if bf16 else 2e-4
            )
            max_error = 0.0
            for a, b in zip(model.parameters(), reference.parameters()):
                max_error = max(max_error, float((a.grad - b.grad).abs().max()))
                torch.testing.assert_close(
                    a.grad, b.grad, atol=0.002 if bf16 else 2e-6, rtol=0.03 if bf16 else 2e-4
                )
            results.append(
                {
                    "bf16": bf16,
                    "microbatch": microbatch,
                    "chunk": chunk,
                    "compiled": compiled,
                    "max_gradient_absolute_error": max_error,
                }
            )
            del ddp, model
    if rank == 0:
        print(json.dumps({"correctness": "passed", "cases": results}), flush=True)
finally:
    dist.destroy_process_group()
