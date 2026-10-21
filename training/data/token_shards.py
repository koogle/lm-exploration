"""Seekable uint16 token records and explicit snapshot verification/conversion.

Each record has a uint32 content-token count, a 32-byte document hash, then
little-endian uint16 content tokens. BOS/EOS remain virtual loader boundaries.
Provenance is stored separately; training never parses it or decompresses it.
"""

import argparse
import gzip
import json
import shutil
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


def convert_snapshot(manifest_path, destination):
    """Verify and convert a legacy gzip snapshot once, preserving its original."""
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("token_format") is not None or manifest.get("schema", 1) != 1:
        raise ValueError("Conversion requires a legacy schema-1 gzip snapshot")
    manifest = verify_snapshot(manifest_path)
    destination.mkdir(parents=True, exist_ok=False)
    for source in manifest["sources"].values():
        for split in source["splits"].values():
            converted = []
            for shard in split["shards"]:
                path = snapshot_file(manifest_path.parent, shard)
                if not path.name.endswith(".jsonl.gz"):
                    raise ValueError("Expected a legacy .jsonl.gz shard")
                target = destination / (path.name.removesuffix(".jsonl.gz") + ".tokens")
                with gzip.open(path, "rt") as handle:
                    converted.append(write_shard(target, (json.loads(line) for line in handle)))
            split["shards"] = converted
    audit = snapshot_file(manifest_path.parent, manifest["audit"])
    shutil.copyfile(audit, destination / audit.name)
    manifest["audit"] = file_entry(destination / audit.name)
    manifest.update(schema=2, token_format=FORMAT, converted_from_sha256=digest(manifest_path))
    # Publish the manifest last; incomplete conversions cannot be used as snapshots.
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest")
    convert = subparsers.add_parser("convert")
    convert.add_argument("manifest")
    convert.add_argument("destination")
    args = parser.parse_args()
    if args.command == "verify":
        verify_snapshot(args.manifest)
    else:
        convert_snapshot(args.manifest, args.destination)


if __name__ == "__main__":
    main()
