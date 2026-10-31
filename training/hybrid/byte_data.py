"""UTF-8 byte streams of the prepared GLM shards, for the byte-level H-Net.

The GLM tokenizer is byte-level BPE without a normalizer, so decoding a shard's
token ids gives back each document's exact UTF-8 bytes. `token_byte_table` maps
every token id to its bytes (added tokens to their text, EOS to the byte-stream
EOS 256); `token_bytes` decodes a token array into a uint16 byte stream.
`ByteMixture` reads the same shards in the same order as `PreparedMixture`, with
windows of `context + 1` bytes. Source weights are the recipe's token weights
converted to bytes with each source's measured bytes per token, so the H-Net sees
the baseline's document mix.
"""

import functools
import json
from pathlib import Path

import numpy as np
from training_data.packed_shards import (
    PreparedMixture,
    heldout_index,
    pack_windows,
    shard_paths,
    wait_for_heldout,
)

from tokenizer import glm

from .hnet import BYTE_EOS


def byte_level_alphabet():
    """GPT-2's byte-to-character map used by the tokenizers ByteLevel pre-tokenizer."""
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    characters = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, characters, strict=True)}


def token_byte_table(tokenizer, eos_id):
    """(uint16 blob, int64 offsets [vocab + 1]): token id i is blob[offsets[i]:offsets[i + 1]]."""
    alphabet = byte_level_alphabet()
    added = {token.content: id_ for id_, token in tokenizer.get_added_tokens_decoder().items()}
    added = {id_: content for content, id_ in added.items()}
    pieces = []
    for id_ in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        if id_ == eos_id:
            pieces.append([BYTE_EOS])
        elif id_ in added:
            pieces.append(list(added[id_].encode()))
        else:
            pieces.append([alphabet[c] for c in tokenizer.id_to_token(id_)])
    offsets = np.zeros(len(pieces) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in pieces])
    blob = np.fromiter((b for p in pieces for b in p), dtype=np.uint16, count=offsets[-1])
    return blob, offsets


def token_bytes(ids, table, block=1 << 24):
    """uint16 byte stream of a token-id array, decoded `block` tokens at a time."""
    blob, offsets = table
    parts = []
    for start in range(0, len(ids), block):
        chunk = np.asarray(ids[start : start + block], dtype=np.int64)
        first, lengths = offsets[chunk], offsets[chunk + 1] - offsets[chunk]
        ends = np.cumsum(lengths)
        index = np.arange(ends[-1] if len(ends) else 0) + np.repeat(
            first - (ends - lengths), lengths
        )
        parts.append(blob[index])
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)


def bytes_per_token(root, recipe, name):
    """Byte-stream units (UTF-8 bytes plus one EOS per document) per token, from the held-out shard."""
    meta = json.loads(
        shard_paths(root, name, heldout_index(recipe["sources"][name]))[2].read_text()
    )
    return (meta["bytes"] + meta["documents"]) / meta["tokens"]


class ByteMixture(PreparedMixture):
    """`PreparedMixture` over the shards' byte streams, in windows of `context + 1` bytes."""

    def __init__(self, recipe, root, context, table, **options):
        super().__init__(recipe, root, **options)
        self.context, self.table = context, table
        weights = {
            name: source["weight"] * bytes_per_token(self.root, recipe, name)
            for name, source in recipe["sources"].items()
        }
        total = sum(weights.values())
        self.weights = {name: weight / total for name, weight in weights.items()}

    def _load(self, path):
        return token_bytes(super()._load(path), self.table)


def heldout_byte_windows(recipe, root, name, context, max_documents, table):
    """Whole held-out documents as bytes packed into windows, with their total UTF-8 bytes."""
    source = recipe["sources"][name]
    bin_path, docs_path, _ = shard_paths(root, name, heldout_index(source))
    lengths = np.load(docs_path)[:max_documents]
    tokens = np.fromfile(bin_path, dtype=np.uint32, count=int(lengths[:, 0].sum()))
    stream = token_bytes(tokens, table)
    total_bytes = int(lengths[:, 1].sum())
    if len(stream) != total_bytes + len(lengths):
        raise ValueError(f"{name}: decoded bytes do not match the shard's recorded lengths")
    return (*pack_windows(stream.astype(np.int64), context), total_bytes)


@functools.cache
def glm_byte_table():
    return token_byte_table(glm.load(), glm.EOS_ID)


def recipe_data(config, data_root):
    """(identity, mixture factory, held-out factory(mixture, context)) for a run config's
    recipe read as bytes; held-out windows wait for each source's held-out shard."""
    recipe = json.loads(Path(config.recipe).read_text())
    root = Path(data_root) / recipe["name"]
    context = config.model["block_size"]

    def mixture():
        return ByteMixture(recipe, root, context, glm_byte_table())

    def heldout(mixture, context):
        wait_for_heldout(recipe, mixture)
        return {
            name: heldout_byte_windows(recipe, mixture.root, name, context, count, mixture.table)
            for name, count in config.eval_documents.items()
        }

    identity = {"recipe": recipe, "tokenizer_sha256": glm.SHA256, "stream": "utf-8 bytes"}
    return identity, mixture, heldout
