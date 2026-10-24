"""Pinned upstream streams and transactional, bounded token staging.

This is preparation, not an eligibility or evaluation-contamination approval.
SQLite is the authoritative cursor/dedup store. Use one writer per output path.
"""

import gzip
import hashlib
import itertools
import json
import re
import sqlite3
import struct
from pathlib import Path

from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen
from .token_shards import FORMAT, file_entry, write_shard

SPLIT_VERSION = "scholarly-parent-v2"
MAX_CHARS = 4_000_000
MAX_LINE_BYTES = 64 * 1024 * 1024
# A small source inside a larger mixture tolerates many repeats (arXiv 2605.12715);
# a recipe can raise its default two passes up to this cap.
MAX_PASSES = 16


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_recipe(path, listing=None):
    """Load a recipe, expanding each source's `files` spec into the explicit shard list.

    Prepared-data identity uses the expanded list, so a spec and the list it expands
    to prepare and reuse the same data.
    """
    recipe = json.loads(Path(path).read_text())
    for source in recipe["sources"]:
        source["files"] = resolve_files(source, listing or hf_glob)
    return validate_recipe(recipe)


# A shard file name ends in `-NNNNN` before its extension, e.g. `org--repo-00003.jsonl.gz`.
SHARD_NAME = re.compile(r"^(?P<stream>.+)-(?P<index>\d+)\.[a-z.]+$")


def hf_glob(source, pattern):
    """Repository-relative paths matching `pattern` at the source's pinned revision."""
    from huggingface_hub import HfFileSystem

    root = f"datasets/{source['repo']}@{source['revision']}/"
    return [path.removeprefix(root) for path in HfFileSystem().glob(root + pattern)]


def resolve_files(source, listing):
    """The ordered shard list for a source's `files`.

    `files` is an explicit list, or one of:

    - `{"pattern": "shard_{:05d}.parquet", "range": [start, stop, step]}`: formats
      each index of `range(start, stop, step)`.
    - `{"glob": "raw/*/*.jsonl.gz", "latest_export": true, "order": "interleave"}`:
      lists matching files at the pinned revision. `latest_export` keeps, for each
      stream (file name without its `-NNNNN` shard number), only the newest parent
      directory. `order` is `"path"` (sorted paths, the default) or `"interleave"`
      (shard number, then file name: every stream's first shard, then every second).
    """
    spec = source["files"]
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        raise ValueError("files must be a list or a file spec")
    if set(spec) == {"pattern", "range"}:
        if len(spec["range"]) != 3 or any(type(v) is not int for v in spec["range"]):
            raise ValueError("A files range is [start, stop, step]")
        return [spec["pattern"].format(index) for index in range(*spec["range"])]
    if "glob" not in spec or set(spec) - {"glob", "latest_export", "order"}:
        raise ValueError("Unknown files spec")
    order = spec.get("order", "path")
    if order not in ("path", "interleave"):
        raise ValueError("files order must be path or interleave")
    paths = sorted(listing(source, spec["glob"]))
    names = {}
    for path in paths:
        match = SHARD_NAME.match(path.rsplit("/", 1)[-1])
        if match is None and (spec.get("latest_export") or order == "interleave"):
            raise ValueError(f"Shard file name has no -NNNNN number: {path}")
        names[path] = match
    if spec.get("latest_export"):
        newest = {}
        for path in paths:
            stream, export = names[path]["stream"], path.rsplit("/", 2)[-2]
            newest[stream] = max(newest.get(stream, export), export)
        paths = [p for p in paths if p.rsplit("/", 2)[-2] == newest[names[p]["stream"]]]
    if order == "interleave":
        paths.sort(key=lambda p: (int(names[p]["index"]), p.rsplit("/", 1)[-1]))
    return paths


def validate_recipe(recipe):
    sources = recipe["sources"]
    if sum(s["weight_percent"] for s in sources) != 100:
        raise ValueError("Source weights must sum to 100")
    if len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Duplicate source names")
    for source in sources:
        if not re.fullmatch(r"[a-z0-9_]+", source["name"]):
            raise ValueError("Unsafe source name")
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Every upstream revision must be a commit SHA")
        if source["format"] not in ("json.gz", "parquet") or not source["files"]:
            raise ValueError("Unsupported or empty source")
        if source["weight_percent"] <= 0:
            raise ValueError("Weights must be positive")
        passes = source.get("max_passes", 2)
        if type(passes) is not int or not 1 <= passes <= MAX_PASSES:
            raise ValueError(f"max_passes must be an integer from 1 to {MAX_PASSES}")
        if source.get("split_by", "document") not in ("document", "row"):
            raise ValueError("split_by must be document or row")
    return recipe


def stream_rows(source, shard, start=0):
    """Reopen pinned shards and replay to the durable row cursor on restart.

    Gzip is sequential HTTP streaming. Parquet uses range reads (no whole-corpus
    download). Closing the generator closes its network/file resources.
    """
    if source["format"] == "json.gz":
        import requests
        from huggingface_hub import get_token

        url = (
            f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{shard}"
        )
        # Private sources need the HF token; public ones ignore it.
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as handle:
                row_index = 0
                while line := handle.readline(MAX_LINE_BYTES + 1):
                    if len(line) > MAX_LINE_BYTES:
                        raise ValueError("Upstream JSON record exceeds memory bound")
                    if row_index >= start:
                        yield row_index, json.loads(line)
                    row_index += 1
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        path = f"datasets/{source['repo']}@{source['revision']}/{shard}"
        with fs.open(path, "rb", block_size=4 * 1024 * 1024) as handle:
            parquet = pq.ParquetFile(handle)
            offset = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                if offset + size <= start:
                    offset += size
                    continue
                for batch in parquet.iter_batches(batch_size=16, row_groups=[group]):
                    for row in batch.to_pylist():
                        if offset >= start:
                            yield offset, row
                        offset += 1


def adapt(source, row, location=None):
    """`location` is the row's (shard file, row index) in the pinned upstream revision.

    Sources with `split_by: "row"` have no document IDs (e.g. pre-shuffled ClimbMix);
    their split is sampled from the location. Exact duplicates in two splits are
    caught by the preparers' content-hash check.
    """
    name = source["name"]
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    if len(text) > MAX_CHARS:
        return None, "oversize"
    if source.get("split_by") == "row":
        if location is None:
            raise ValueError("Row-sampled sources need the row's shard and index")
        document = f"{location[0]}:{location[1]}"
    else:
        document = row.get("id") or row.get("url")
    parent = document
    if not document:
        return None, "missing_document_or_parent"
    parent = f"{name}:{parent}"
    bucket = int(hashlib.sha256(f"{SPLIT_VERSION}:{parent}".encode()).hexdigest(), 16) % 10
    split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
    provenance = {k: v for k, v in row.items() if k not in ("text", "content")}
    return {
        "text": text,
        "document": str(document),
        "parent": parent,
        "split": split,
        "provenance": provenance,
    }, None


class Stager:
    def __init__(self, folder, recipe, tokenizer_folder=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe = recipe
        self.tokenizer = (
            load_frozen() if tokenizer_folder is None else load_frozen(tokenizer_folder)
        )
        self.db = sqlite3.connect(self.folder / "state.sqlite")
        # Hard local staging bound; SQLite rolls back a transaction that exceeds it.
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        self.db.execute(f"PRAGMA max_page_count={8 * 1024**3 // page_size}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS cursors (source TEXT PRIMARY KEY, shard INTEGER, row INTEGER);
            CREATE TABLE IF NOT EXISTS documents (
                hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
                provenance TEXT, tokens BLOB, count INTEGER, conflict INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS decisions (
                source TEXT, reason TEXT, count INTEGER, PRIMARY KEY(source, reason));
            CREATE TABLE IF NOT EXISTS audit (
                source TEXT, shard INTEGER, row INTEGER, reason TEXT,
                content_hash TEXT, parent TEXT, PRIMARY KEY(source, shard, row));
        """)
        identity = canonical(
            {
                "schema": 1,
                "recipe": recipe,
                "tokenizer": TOKENIZER_SHA256,
                "split": SPLIT_VERSION,
                "max_chars": MAX_CHARS,
                "implementation_sha256": digest(__file__),
                "tokenizer_loader_sha256": digest(Path(__file__).with_name("subword.py")),
                "token_shards_sha256": digest(Path(__file__).with_name("token_shards.py")),
            }
        )
        existing = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
        if existing and existing[0] != identity:
            self.db.close()
            raise ValueError("Preparation identity changed; use a fresh versioned path")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (identity,))
        self.db.commit()
        self.identity = json.loads(identity)

    def close(self):
        self.db.close()

    def decision(self, source, reason):
        self.db.execute(
            """INSERT INTO decisions VALUES (?, ?, 1)
            ON CONFLICT(source,reason) DO UPDATE SET count=count+1""",
            (source, reason),
        )

    def prepare(self, source_name, max_records, reader=stream_rows):
        if max_records < 1:
            raise ValueError("max_records must be positive")
        source = next(s for s in self.recipe["sources"] if s["name"] == source_name)
        cursor = self.db.execute(
            "SELECT shard,row FROM cursors WHERE source=?", (source_name,)
        ).fetchone() or (0, 0)
        scanned = 0
        for shard_index in range(cursor[0], len(source["files"])):
            start = cursor[1] if shard_index == cursor[0] else 0
            rows = reader(source, source["files"][shard_index], start)
            try:
                for row_index, row in rows:
                    with self.db:
                        document, reason = adapt(
                            source, row, (source["files"][shard_index], row_index)
                        )
                        content_hash = None
                        if document is not None:
                            content_hash = hashlib.sha256(document["text"].encode()).hexdigest()
                            previous = self.db.execute(
                                "SELECT split FROM documents WHERE hash=?", (content_hash,)
                            ).fetchone()
                            if previous:
                                reason = "exact_duplicate"
                                if previous[0] != document["split"]:
                                    self.db.execute(
                                        "UPDATE documents SET conflict=1 WHERE hash=?",
                                        (content_hash,),
                                    )
                                    reason = "cross_split_duplicate"
                            else:
                                ids = (
                                    []
                                    if document["split"] == "test"
                                    else self.tokenizer.encode(
                                        document["text"], add_special_tokens=False
                                    ).ids
                                )
                                # Literal spellings of special tokens must not become control IDs.
                                if any(t < FIRST_CONTENT_TOKEN_ID for t in ids):
                                    reason = "special_token_in_content"
                                provenance = {
                                    "document_id": document["document"],
                                    "upstream": document["provenance"],
                                    "repo": source["repo"],
                                    "revision": source["revision"],
                                    "shard": source["files"][shard_index],
                                    "row": row_index,
                                }
                                if reason is None:
                                    self.db.execute(
                                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                                        (
                                            content_hash,
                                            source_name,
                                            document["split"],
                                            document["parent"],
                                            canonical(provenance),
                                            struct.pack(f"<{len(ids)}H", *ids),
                                            len(ids),
                                        ),
                                    )
                                    reason = "excluded_test" if not ids else "accepted"
                        self.decision(source_name, reason)
                        self.db.execute(
                            "INSERT INTO audit VALUES (?,?,?,?,?,?)",
                            (
                                source_name,
                                shard_index,
                                row_index,
                                reason,
                                content_hash,
                                document["parent"] if document else None,
                            ),
                        )
                        self.db.execute(
                            "INSERT OR REPLACE INTO cursors VALUES (?,?,?)",
                            (source_name, shard_index, row_index + 1),
                        )
                    scanned += 1
                    if scanned >= max_records:
                        return scanned
            finally:
                if hasattr(rows, "close"):
                    rows.close()
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO cursors VALUES (?,?,0)", (source_name, shard_index + 1)
                )
        return scanned

    def export(self, destination, documents_per_shard=128):
        """Freeze a new immutable snapshot; never append to an existing snapshot."""
        if documents_per_shard < 1:
            raise ValueError("documents_per_shard must be positive")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema": 2,
            "token_format": FORMAT,
            "implementation_sha256": self.identity["implementation_sha256"],
            "tokenizer_sha256": TOKENIZER_SHA256,
            "recipe_sha256": hashlib.sha256(canonical(self.recipe).encode()).hexdigest(),
            "split_policy": SPLIT_VERSION,
            "training_ready": False,
            "pending": [
                "reconcile tokenizer mixture-split-v1 parent assignments",
                "global near deduplication and cross-source parent reconciliation",
                "benchmark contamination and eligibility/provenance review",
                "full usable capacity measurement",
            ],
            "packing": "BOS + document + EOS; document-local windows; padding labels -100",
            "max_passes": 2,
            "sources": {},
            "decisions": {},
            "cursors": {},
        }
        for source in self.recipe["sources"]:
            name = source["name"]
            entry = {
                "weight_percent": source["weight_percent"],
                "repo": source["repo"],
                "revision": source["revision"],
                "splits": {},
            }
            for split in ("train", "validation"):
                rows = self.db.execute(
                    """SELECT hash,parent,provenance,tokens,count FROM documents
                    WHERE source=? AND split=? AND conflict=0 ORDER BY hash""",
                    (name, split),
                )
                shards, tokens, docs = [], 0, 0
                while first := next(rows, None):
                    path = destination / f"{name}-{split}-{len(shards):06d}.tokens"
                    batch = itertools.chain(
                        [first], itertools.islice(rows, documents_per_shard - 1)
                    )

                    def records(batch=batch):
                        nonlocal tokens, docs
                        for content_hash, parent, provenance, data, count in batch:
                            tokens += count + 1  # EOS bears loss; BOS is input only.
                            docs += 1
                            yield {
                                "hash": content_hash,
                                "parent": parent,
                                "provenance": json.loads(provenance),
                                "ids": list(struct.unpack(f"<{count}H", data)),
                            }

                    shards.append(write_shard(path, records()))
                target = 20_000_000_000 * source["weight_percent"] // 100
                entry["splits"][split] = {
                    "documents": docs,
                    "loss_tokens": tokens,
                    "shards": shards,
                }
                if split == "train":
                    entry["target_20b"] = target
                    entry["observed_max_exposures_two_passes"] = tokens * 2
                    entry["observed_capacity_sufficient"] = tokens * 2 >= target
            manifest["sources"][name] = entry
        for source, reason, count in self.db.execute("SELECT * FROM decisions"):
            manifest["decisions"].setdefault(source, {})[reason] = count
        for source, shard, row in self.db.execute("SELECT * FROM cursors"):
            manifest["cursors"][source] = {"shard": shard, "row": row}
        audit_path = destination / "audit.jsonl.gz"
        with (
            audit_path.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as output,
        ):
            for source, shard, row, reason, content_hash, parent in self.db.execute(
                "SELECT * FROM audit ORDER BY source,shard,row"
            ):
                pin = next(s for s in self.recipe["sources"] if s["name"] == source)
                output.write(
                    (
                        canonical(
                            {
                                "source": source,
                                "repo": pin["repo"],
                                "revision": pin["revision"],
                                "shard": pin["files"][shard],
                                "row": row,
                                "decision": reason,
                                "content_hash": content_hash,
                                "parent": parent,
                            }
                        )
                        + "\n"
                    ).encode()
                )
        manifest["audit"] = file_entry(audit_path)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
