"""Load trusted, hash-verified model sources without replacing installed packages."""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path


def load_archived_model(source, expected_hashes):
    source = Path(source)
    packages = ("model_10m", "model_100m")
    if set(expected_hashes) != set(packages):
        raise ValueError("Both archived model source hashes are required")
    for package in packages:
        path = source / package / "model.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[package]:
            raise ValueError(f"Archived {package} source checksum mismatch")

    def belongs(name):
        return any(name == package or name.startswith(package + ".") for package in packages)

    previous = {name: module for name, module in sys.modules.copy().items() if belongs(name)}
    try:
        for name in previous:
            del sys.modules[name]
        for package in packages:
            module = types.ModuleType(package)
            module.__path__ = [str(source / package)]
            sys.modules[package] = module
            name = package + ".model"
            spec = importlib.util.spec_from_file_location(name, source / package / "model.py")
            model = importlib.util.module_from_spec(spec)
            sys.modules[name] = model
            spec.loader.exec_module(model)
            module.model = model
        return model.GPT, model.ModelConfig
    finally:
        for name in list(sys.modules):
            if belongs(name):
                del sys.modules[name]
        sys.modules.update(previous)
