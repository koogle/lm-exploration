"""Bounded CPU preparation and optional asynchronous CUDA copies.

Factory construction, iteration and closing occur on the producer thread. This
allows SQLite-backed loaders to keep their connection confined to one thread.
Only acknowledge() advances the consumer cursor saved in a training checkpoint.
"""

import copy
import queue
import threading
import time


class PrefetchedBatches:
    def __init__(self, factory, batch_size, *, depth=8, state=None, device=None):
        if not 1 <= depth <= 64 or batch_size < 1:
            raise ValueError("Invalid prefetch depth or batch size")
        self.factory, self.batch_size, self.depth = factory, batch_size, depth
        self.device = device
        self.initial = state
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.initialized = threading.Event()
        self.source = None
        self.error = None
        self.pending = None
        self.wait_seconds = 0.0
        self.batches = 0
        self.thread = threading.Thread(target=self._produce, daemon=True, name="batch-preparation")
        self.thread.start()
        if not self.initialized.wait(300):
            self.stop.set()
            raise TimeoutError("Loader initialization exceeded five minutes")
        if self.error:
            raise self.error

    def _cursor(self):
        # Live cursor_state returns fresh dictionaries containing immutable
        # document-token tuples. Recopying million-token tuples per batch would
        # defeat prefetch; full ledgers are serialized only at checkpoints.
        if hasattr(self.source, "cursor_state"):
            return self.source.cursor_state()
        return copy.deepcopy(self.source.state_dict())

    def _produce(self):
        stream = None
        try:
            self.source = self.factory()
            if self.initial is not None:
                self.source.load_state_dict(self.initial)
            self.committed = self._cursor()
            self.identity = self.source.identity
            self.sources = self.source.sources
            self.manifest = self.source.manifest
            stream = None
            if self.device is not None:
                import torch

                torch.cuda.set_device(self.device)
                stream = torch.cuda.Stream(device=self.device)
            self.initialized.set()
            while not self.stop.is_set():
                # Do not prepare an unbounded extra packet behind a full queue.
                if self.queue.full():
                    self.stop.wait(0.005)
                    continue
                try:
                    x, y = self.source.next_batch(self.batch_size, "cpu")
                except StopIteration:
                    self._put(("end", None))
                    return
                cursor = self._cursor()
                packet = {"x": x, "y": y, "cursor": cursor, "event": None}
                if stream is not None:
                    import torch

                    x, y = x.pin_memory(), y.pin_memory()
                    with torch.cuda.stream(stream):
                        packet.update(
                            x=x.to(self.device, non_blocking=True),
                            y=y.to(self.device, non_blocking=True),
                            host=(x, y),
                        )
                        packet["event"] = stream.record_event()
                if not self._put(("batch", packet)):
                    return
        except Exception as exc:  # noqa: BLE001 -- forward producer failures to the consumer
            self.error = exc
            self.initialized.set()
            self._put(("error", exc))
        finally:
            if stream is not None:
                stream.synchronize()
            if self.source is not None and hasattr(self.source, "close"):
                self.source.close()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def next_batch(self, batch_size, device="cpu"):
        if batch_size != self.batch_size or self.pending is not None:
            raise ValueError("Acknowledge the outstanding batch before requesting another")
        started = time.monotonic()
        while True:
            try:
                kind, payload = self.queue.get(timeout=0.1)
                break
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError("Preparation producer stopped unexpectedly")
        self.wait_seconds += time.monotonic() - started
        if kind == "error":
            raise payload
        if kind == "end":
            raise StopIteration
        self.pending = payload
        if payload["event"] is not None:
            import torch

            current = torch.cuda.current_stream(self.device)
            current.wait_event(payload["event"])
            payload["x"].record_stream(current)
            payload["y"].record_stream(current)
        x, y = payload["x"], payload["y"]
        if self.device is None and device != "cpu":
            x, y = x.to(device), y.to(device)
        return x, y

    def acknowledge(self):
        if self.pending is None:
            raise ValueError("No outstanding batch")
        self.committed = self.pending["cursor"]
        self.pending = None
        self.batches += 1

    @property
    def exposures(self):
        return self.committed["loss_token_exposures"]

    def state_dict(self):
        if hasattr(self.source, "state_at"):
            return self.source.state_at(self.committed)
        return copy.deepcopy(self.committed)

    def load_state_dict(self, state):
        # Used for an untrained token-ceiling batch. Reconstructing discards all
        # speculative preparation and cannot skip data on a later continuation.
        self.close()
        self.__init__(
            self.factory, self.batch_size, depth=self.depth, state=state, device=self.device
        )

    def stats(self):
        return {
            "data_wait_seconds": self.wait_seconds,
            "consumed_batches": self.batches,
            "ready_batches": self.queue.qsize(),
            "buffer_capacity": self.depth,
        }

    def close(self):
        self.stop.set()
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            raise TimeoutError("Producer did not stop within two minutes")
