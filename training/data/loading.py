"""Small, in-memory UTF-8 datasets and a deterministic character tokenizer.

Vocabulary is taken from the whole corpus. The last 10% is held out; random
contiguous windows never cross the split. This is not a large-corpus pipeline.
"""

import hashlib
from pathlib import Path

import torch


class CharacterTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        self.indices = {character: index for index, character in enumerate(self.characters)}

    def encode(self, text):
        unknown = set(text) - self.indices.keys()
        if unknown:
            raise ValueError(f"characters absent from vocabulary: {sorted(unknown)!r}")
        return [self.indices[character] for character in text]

    def decode(self, tokens):
        return "".join(self.characters[index] for index in tokens)

    def config(self):
        return {"type": "character", "characters": self.characters}


def load_data(path, block_size):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")  # No newline normalization: hash identifies exact bytes.
    tokenizer = CharacterTokenizer(sorted(set(text)))
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = len(tokens) * 9 // 10
    if min(split, len(tokens) - split) < block_size + 1:
        raise ValueError(
            "both the 90% train and 10% validation splits need block_size + 1 characters"
        )
    fingerprint = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "characters": len(text),
        "train_tokens": split,
        "validation_tokens": len(tokens) - split,
    }
    return tokenizer, tokens[:split], tokens[split:], fingerprint


def get_batch(data, batch_size, block_size, generator, device):
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)
