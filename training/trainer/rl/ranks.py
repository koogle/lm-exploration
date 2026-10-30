"""The RL loop's processes: one per GPU, the same prompts on each, each sampling its share.

`Ranks` wraps the collectives the loop needs (a process group initialized by the caller):
`shard` takes this rank's share of a list, `gather` concatenates every rank's lists,
`sum` adds numbers across ranks, `average_gradients` replaces each gradient by its mean
over ranks, and `broadcast` sends rank 0's value. `Ranks()` is the single process.

`run_ranks` starts one process per GPU. The launching process keeps what the ranks
cannot hold themselves (the code sandbox, the volume, the stop request) and serves it to
them as `Services` over an authenticated local socket (`serve`, `connect`).
"""


class Ranks:
    def __init__(self, rank=0, world=1):
        self.rank, self.world = rank, world
        self.main = rank == 0

    def shard(self, items):
        if len(items) % self.world:
            raise ValueError(f"{len(items)} items do not split over {self.world} ranks")
        return items[self.rank :: self.world]

    def gather(self, items):
        """Every rank's `items` (lists), concatenated in rank order."""
        if self.world == 1:
            return list(items)
        import torch.distributed as dist

        parts = [None] * self.world
        dist.all_gather_object(parts, list(items))
        return [item for part in parts for item in part]

    def unshard(self, items):
        """Every rank's shard of a list back in the original order (`shard` inverted)."""
        parts = self.gather([items])
        return [part[i] for i in range(len(parts[0])) for part in parts]

    def sum(self, value):
        if self.world == 1:
            return value
        return sum(self.gather([value]))

    def broadcast(self, value):
        if self.world == 1:
            return value
        import torch.distributed as dist

        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def average_gradients(self, model):
        if self.world == 1:
            return
        import torch
        import torch.distributed as dist

        # One all-reduce per tensor: no flat copy of every gradient at once.
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= self.world

    def barrier(self):
        if self.world > 1:
            import torch.distributed as dist

            dist.barrier()


class Services:
    """What only the launching process can do, for the rank processes: check code in the
    sandbox (`run`), commit the run volume and read the stop request."""

    def __init__(self, sandbox, commit, should_stop):
        self.sandbox, self._commit, self._should_stop = sandbox, commit, should_stop

    def run(self, pairs):
        return self.sandbox.run(pairs)

    def commit(self):
        self._commit()

    def should_stop(self):
        return bool(self._should_stop())


def serve(services):
    """Serve `services` on a local socket from a thread; (address, key) for `connect`."""
    import secrets
    import threading
    from multiprocessing.managers import BaseManager

    class Manager(BaseManager):
        pass

    Manager.register("services", callable=lambda: services)
    key = secrets.token_bytes(32)
    server = Manager(address=("127.0.0.1", 0), authkey=key).get_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.address, key


def connect(address, key):
    """A proxy of the launching process's `Services`."""
    from multiprocessing.managers import BaseManager

    class Manager(BaseManager):
        pass

    Manager.register("services")
    manager = Manager(address=address, authkey=key)
    manager.connect()
    return manager.services()


def run_ranks(world, target, *args):
    """`target(ranks, *args)` in one process per GPU with an NCCL process group (gloo
    without CUDA); rank 0's return value."""
    import json
    import tempfile
    from pathlib import Path

    import torch.multiprocessing as mp

    result = Path(tempfile.mkdtemp()) / "rank0.json"
    mp.spawn(_rank, args=(world, target, args, str(result)), nprocs=world, join=True)
    return json.loads(result.read_text())


def _rank(rank, world, target, args, result):
    import json
    import os
    from pathlib import Path

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    else:  # CPU processes (tests)
        dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        value = target(Ranks(rank, world), *args)
        if rank == 0:
            Path(result).write_text(json.dumps(value))
    finally:
        dist.destroy_process_group()
