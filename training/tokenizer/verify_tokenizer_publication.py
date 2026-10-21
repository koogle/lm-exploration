"""Verify local or commit-pinned Hub tokenizers through the standard loading API."""

import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def verify(location, revision=None):
    root = Path(location)
    if not root.is_dir():
        if not revision:
            raise ValueError("Remote verification requires an immutable commit revision")
        root = Path(snapshot_download(location, revision=revision))
    manifest = json.loads((root / "publication-manifest.json").read_text())
    variants = {".": manifest["variants"][manifest["default_variant"]]}
    variants.update({f"variants/{name}": sha for name, sha in manifest["variants"].items()})
    probes = [
        "def square(x):\r\n\treturn x ** 2\r\n",
        "∫₀¹ x² dx = ⅓",
        "naïve café 中文 العربية 😀",
        "\x00\t \r\n",
        "e\u0301 ≠ é",
        "<|bos|>text<|eos|>",
    ]
    for name, expected in variants.items():
        path = root / name
        if hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        raw = tokenizers.Tokenizer.from_file(str(path / "tokenizer.json"))
        if [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        ] != [0, 1, 2, 3]:
            raise ValueError(f"Special IDs changed: {name}")
        requested = int((manifest["default_variant"] if name == "." else name).rsplit("-", 1)[1])
        if len(tokenizer) != requested:
            raise ValueError(f"Vocabulary size changed: {name}")
        for text in probes:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids != raw.encode(text, add_special_tokens=False).ids:
                raise ValueError(f"Wrapper token IDs differ: {name}")
            if (
                tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                != text
            ):
                raise ValueError(f"Wrapper round trip failed: {name}")
            if tokenizer.encode(text, add_special_tokens=True) != ids:
                raise ValueError(f"Unexpected automatic boundaries: {name}")
    return {
        "location": location,
        "revision": revision,
        "status": "passed",
        "default_and_variants_checked": len(variants),
        "probes_per_variant": len(probes),
        "checks": [
            "artifact_hashes",
            "special_ids",
            "vocabulary_size",
            "token_id_equivalence",
            "exact_roundtrips",
            "no_automatic_boundaries",
        ],
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.location, args.revision)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
