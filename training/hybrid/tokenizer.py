"""The GLM tokenizer adopted by the hybrid family.

zai-org/GLM-4.7-Flash's tokenizer.json, pinned by revision and SHA-256. It is
byte-identical to zai-org/GLM-5.3-Flash's at the revision below (checked
2026-10-25). Documents end with `<|endoftext|>`; the chat, reasoning and tool
special tokens are GLM's own. The model's embedding table pads the 154,856
tokenizer entries to 154,880 rows.
"""

import hashlib
from pathlib import Path

SOURCE = {
    "repo": "zai-org/GLM-4.7-Flash",
    "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "file": "tokenizer.json",
}
IDENTICAL_TO = {
    "repo": "zai-org/GLM-5.3-Flash",
    "revision": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
}
SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_SIZE = 154_856
VOCAB_SIZE = 154_880
EOS_ID = 154_820
SPECIAL_TOKENS = {
    "<|endoftext|>": 154_820,
    "[gMASK]": 154_822,
    "<sop>": 154_824,
    "<|system|>": 154_826,
    "<|user|>": 154_827,
    "<|assistant|>": 154_828,
    "<|observation|>": 154_829,
    "<|code_prefix|>": 154_838,
    "<|code_middle|>": 154_839,
    "<|code_suffix|>": 154_840,
    "<think>": 154_841,
    "</think>": 154_842,
    "<tool_call>": 154_843,
    "</tool_call>": 154_844,
    "<tool_response>": 154_845,
    "</tool_response>": 154_846,
    "<arg_key>": 154_847,
    "</arg_key>": 154_848,
    "<arg_value>": 154_849,
    "</arg_value>": 154_850,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_file(local_dir=None):
    """Download the pinned tokenizer.json (cached) and verify its hash."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        SOURCE["repo"], SOURCE["file"], revision=SOURCE["revision"], local_dir=local_dir
    )
    if sha256(path) != SHA256:
        raise ValueError(f"{path} does not match the pinned GLM tokenizer hash")
    return Path(path)


def load(path=None):
    """The verified GLM tokenizer, with its special-token ids checked."""
    from tokenizers import Tokenizer

    path = Path(path) if path else tokenizer_file()
    if sha256(path) != SHA256:
        raise ValueError(f"{path} does not match the pinned GLM tokenizer hash")
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != TOKENIZER_SIZE:
        raise ValueError("Unexpected GLM tokenizer size")
    for token, expected in SPECIAL_TOKENS.items():
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"GLM special token {token} is not id {expected}")
    return tokenizer
