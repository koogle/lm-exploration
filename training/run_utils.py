"""Run infrastructure: checkpoint I/O, RNG state, provenance and clean stopping."""

import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


def rng_state(batch_generator):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "batches": batch_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, batch_generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    batch_generator.set_state(state["batches"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def atomic_write(path, writer):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(
        path,
        lambda stream: stream.write(
            (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        ),
    )


def load_checkpoint(path):
    # Only load trusted checkpoints: this restores Python RNG objects as well.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return checkpoint


def runtime_provenance(device, model_source):
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    files = {
        "model": Path(model_source),
        "training": root / "train.py",
        "data": root / "data" / "loading.py",
        "runtime": Path(__file__),
        "dependencies": root / "uv.lock",
    }
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": device,
        "git_commit": commit,
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
            if path.is_file()
        },
        "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }


@contextmanager
def stop_at_step_boundary():
    requested = [False]

    def request_stop(signum, frame):
        requested[0] = True  # Never raise while optimizer weights are being updated.

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
