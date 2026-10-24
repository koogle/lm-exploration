"""Training-step throughput with the fixed loss versus F.cross_entropy, under torchrun.

    torchrun --nproc_per_node=N -m trainer.loss_benchmark <recipe.json> [steps]

Runs the loop's own pieces (model, DDP options, optimizer, backward_batch, clipping)
for full optimizer steps of the recipe on random tokens, once per loss, from the same
initialization. Data loading and checkpointing are excluded. Rank 0 prints one JSON
line. No pretraining is launched.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path


def main(recipe, steps=20, warmup=3):
    import torch
    import torch.distributed as dist
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from training_data.subword import VOCAB_SIZE

    from distributed.config import DistributedConfig
    from trainer import loss as loss_module
    from trainer.loop import backward_batch, make_optimizer

    fixed = loss_module.cross_entropy

    def standard(logits, targets, **kwargs):
        # What autocast runs for F.cross_entropy on bf16 logits.
        return F.cross_entropy(logits.float(), targets, **kwargs)

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    config = DistributedConfig(**{**json.loads(Path(recipe).read_text()), "world_size": world})
    local_windows = config.global_windows // world
    results = []
    try:
        for name, fn in (("torch_cross_entropy", standard), ("cuda_kernel", fixed)):
            # backward_batch imports trainer.loss.cross_entropy when called.
            loss_module.cross_entropy = fn
            torch.manual_seed(config.seed)
            model = GPT(
                ModelConfig(
                    vocab_size=VOCAB_SIZE,
                    block_size=config.context,
                    n_layer=config.n_layer,
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                )
            ).cuda()
            ddp = DDP(
                model, device_ids=[rank], gradient_as_bucket_view=True, broadcast_buffers=False
            )
            optimizer = make_optimizer(model, config)
            generator = torch.Generator(device="cuda").manual_seed(config.seed + rank)
            torch.cuda.reset_peak_memory_stats()
            durations, losses = [], []
            for step in range(warmup + steps):
                x, y = (
                    torch.randint(
                        0, VOCAB_SIZE, (local_windows, config.context), device="cuda",
                        generator=generator,
                    )
                    for _ in range(2)
                )  # fmt: skip
                count = config.global_windows * config.context
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dist.barrier()
                start = time.perf_counter()
                total = backward_batch(
                    ddp, x, y, count, config.microbatch, world,
                    crop_padding=config.crop_padding, loss_chunk_size=config.loss_chunk_size,
                )  # fmt: skip
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm, foreach=True
                )
                optimizer.step()
                torch.cuda.synchronize()
                dist.all_reduce(total)
                if step >= warmup:
                    durations.append(time.perf_counter() - start)
                    losses.append(float(total) / count)
            median = statistics.median(durations)
            results.append(
                {
                    "loss": name,
                    "steps": steps,
                    "median_step_seconds": median,
                    "min_step_seconds": min(durations),
                    "max_step_seconds": max(durations),
                    "tokens_per_second": config.global_windows * config.context / median,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                }
            )
            del ddp, model, optimizer
            torch.cuda.empty_cache()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "recipe": Path(recipe).name,
                        "world_size": world,
                        "global_windows": config.global_windows,
                        "microbatch": config.microbatch,
                        "context": config.context,
                        "optimizer": config.optimizer,
                        "results": results,
                    }
                ),
                flush=True,
            )
    finally:
        loss_module.cross_entropy = fixed
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
