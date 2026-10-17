"""Immutable tokenizer contract for scholarly/code data and future checkpoints."""

import hashlib
import json
from pathlib import Path

TOKENIZER_SHA256 = "add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9"
VOCAB_SIZE = 32768
DEFAULT_FOLDER = Path(__file__).resolve().parents[1] / "tokenizer/structured-code-v1"


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
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError("Unexpected vocabulary")
    for index, name in enumerate(("pad", "bos", "eos", "unk")):
        if tokenizer.token_to_id(f"<|{name}|>") != index:
            raise ValueError("Unexpected special token IDs")
    return tokenizer
