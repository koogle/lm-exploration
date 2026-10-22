"""Seekable uint16 token records and explicit snapshot verification.

Each record has a uint32 content-token count, a 32-byte document hash, then
little-endian uint16 content tokens. BOS/EOS remain virtual loader boundaries.
Provenance is stored separately; training never parses it or decompresses it.
"""

import argparse
import gzip
import json
import struct
from pathlib import Path

from .subword import FIRST_CONTENT_TOKEN_ID, VOCAB_SIZE, digest

FORMAT = "uint16-documents-v1"
MAGIC = b"LMTOK01\n"
HEADER = struct.Struct("<I32s")


def snapshot_file(root, entry):
    name = entry["path"]
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Invalid snapshot path")
    path = Path(root) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or nonregular snapshot file: {name}")
    if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Snapshot file size mismatch: {name}")
    return path


def file_entry(path):
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)}


def write_shard(path, rows):
    """Write once during preparation; retain metadata without duplicate token IDs."""
    path = Path(path)
    provenance = path.with_suffix(".provenance.jsonl.gz")
    with (
        path.open("xb") as output,
        provenance.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as metadata,
    ):
        output.write(MAGIC)
        for row in rows:
            ids = row["ids"]
            if not ids or any(
                type(t) is not int or not FIRST_CONTENT_TOKEN_ID <= t < VOCAB_SIZE for t in ids
            ):
                raise ValueError("Invalid content token IDs")
            content_hash = bytes.fromhex(row["hash"])
            if len(content_hash) != 32:
                raise ValueError("Expected SHA-256 document hash")
            output.write(HEADER.pack(len(ids), content_hash))
            output.write(struct.pack(f"<{len(ids)}H", *ids))
            record = {k: v for k, v in row.items() if k != "ids"}
            record["content_tokens"] = len(ids)
            metadata.write((json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return {**file_entry(path), "provenance": file_entry(provenance)}


def verify_snapshot(manifest_path):
    """Explicit full-byte verification, outside the training startup path."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entries = [manifest["audit"]]
    for source in manifest["sources"].values():
        for split in source["splits"].values():
            for shard in split["shards"]:
                entries.append(shard)
                if "provenance" in shard:
                    entries.append(shard["provenance"])
    for entry in entries:
        path = snapshot_file(manifest_path.parent, entry)
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {path.name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest")
    args = parser.parse_args()
    verify_snapshot(args.manifest)


if __name__ == "__main__":
    main()
