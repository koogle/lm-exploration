"""Immutable tokenizer contract for scholarly/code data and future checkpoints."""

import hashlib
import json
from pathlib import Path

TOKENIZER_SHA256 = "2a6bd2343ab29e3955dd7a5e17330e706afc8c650618b1339ceedab5e77fde04"
DEFAULT_FOLDER = Path(__file__).resolve().parents[1] / "tokenizer/scholarly-code-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_frozen(folder=DEFAULT_FOLDER):
    from tokenizers import Tokenizer

    folder = Path(folder)
    manifest = json.loads((folder / "freeze-manifest.json").read_text())
    if manifest["files"]["tokenizer.json"] != TOKENIZER_SHA256:
        raise ValueError("Unexpected tokenizer identity")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f"Frozen artifact failed verification: {name}")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 16384:
        raise ValueError("Unexpected vocabulary")
    for index, name in enumerate(("pad", "bos", "eos", "unk")):
        if tokenizer.token_to_id(f"<|{name}|>") != index:
            raise ValueError("Unexpected special token IDs")
    return tokenizer
