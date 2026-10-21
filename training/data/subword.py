"""Immutable tokenizer contract for Simple LLM data and future checkpoints."""

import hashlib
import json
from pathlib import Path

TOKENIZER_SHA256 = "add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9"
VOCAB_SIZE = 32768

# Vocabulary IDs fixed by the frozen tokenizer artifact.
PAD_TOKEN_ID = 0  # Fill unused input positions.
BOS_TOKEN_ID = 1  # Mark the beginning of a document.
EOS_TOKEN_ID = 2  # Mark the end of a document; this is a prediction target.
UNK_TOKEN_ID = 3  # Reserved unknown-token marker.
SPECIAL_TOKEN_IDS = {
    "<|pad|>": PAD_TOKEN_ID,
    "<|bos|>": BOS_TOKEN_ID,
    "<|eos|>": EOS_TOKEN_ID,
    "<|unk|>": UNK_TOKEN_ID,
}
FIRST_CONTENT_TOKEN_ID = 4  # IDs below this are reserved, not document content.

# Loss-label sentinel, NOT a vocabulary ID. PyTorch ignores these targets.
LOSS_IGNORE_INDEX = -100
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
    for spelling, expected_id in SPECIAL_TOKEN_IDS.items():
        if tokenizer.token_to_id(spelling) != expected_id:
            raise ValueError("Unexpected special token IDs")
    return tokenizer
