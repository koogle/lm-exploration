"""Source and runtime provenance recorded in each run's config.json."""

import hashlib
import platform
import subprocess
from pathlib import Path

import torch


def runtime_provenance(device, model_source):
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    files = {
        "model": Path(model_source),
        "training": root / "trainer" / "loop.py",
        "training_config": root / "trainer" / "config.py",
        "checkpoints": root / "trainer" / "checkpoints.py",
        "sampling": root / "trainer" / "sampling.py",
        "cli": root / "cli.py",
        "data": root / "data" / "loading.py",
        "provenance": Path(__file__),
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
