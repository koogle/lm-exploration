"""Immutable EU raw snapshots and explicit optimizer-preserving domain continuation.

This is a new training phase, not an exact replay of the old data stream. Strict
ordinary resume checks are retained; only this explicit migration resets counters
and the data cursor while preserving weights, Adam moments and parent provenance.
"""

import hashlib
import json
import re
import sqlite3
import unicodedata
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training_data.subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, digest, load_frozen


def family(row):
    """Group language/format variants and revisions before assigning holdouts."""
    provenance = json.loads(row.get("provenance_json") or "{}")
    inventory = provenance.get("inventory_record", {})
    identity = inventory.get("source_document_id") or row["document_id"]
    url = row["url"].lower().split("?")[0]
    # ECB annual reports appear both as translations and as PDF/HTML versions.
    if "annual_report" in row["collection_id"]:
        year = re.search(r"(?:19|20)\d{2}", row.get("title", ""))
        if year:
            identity = "annual-report-" + year[0]
    if row["source_id"] == "ema":
        # Language versions and several EPAR document types share the medicine.
        match = re.search(r"/medicines/[^/]+/[^/]+/([^/]+)", url)
        filename = re.search(r"/([^/]+?)-epar(?:-|\.)", url)
        if match or filename:
            identity = "medicine-" + (match or filename)[1]
    return row["source_id"] + ":" + str(identity)


def split_for(parent):
    value = int(hashlib.sha256(parent.encode()).hexdigest()[:8], 16) % 100
    return "test" if value == 0 else "validation" if value < 3 else "train"


def normalize(text):
    # Preserve paragraphs/code; only canonicalize Unicode and line endings.
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def prepare_snapshot(folder, datasets, *, commit=lambda: None):
    """Fetch only Parquet text shards, select latest records, exact-dedup and count.

    Frozen source revisions + code hash identify the reusable snapshot. Download
    completion and selected rows are transactional, so interrupted prep resumes.
    Token IDs are counted in bounded batches and discarded; training uses the
    existing asynchronous just-in-time raw-text tokenizer.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {
        "datasets": datasets,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "implementation_sha256": digest(__file__),
        "split": "eu-family-sha256-97-2-1-v1",
    }
    identity_path = folder / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Preparation identity changed; use a new snapshot")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if digest(folder / "raw.sqlite") != manifest["database"]["sha256"]:
            raise ValueError("Completed snapshot checksum mismatch")
        return manifest
    db = sqlite3.connect(folder / "staging.sqlite")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS files (repo TEXT, path TEXT, PRIMARY KEY(repo,path));
        CREATE TABLE IF NOT EXISTS latest (
            id TEXT PRIMARY KEY, retrieved TEXT, hash TEXT, parent TEXT,
            institution TEXT, provenance TEXT, text BLOB);
    """)
    api = HfApi()
    files = []
    for spec in datasets:
        names = api.list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])
        files.extend(
            (spec, n) for n in sorted(names) if n.startswith("data/") and n.endswith(".parquet")
        )
    completed = set(db.execute("SELECT repo,path FROM files"))
    pending = [(s, n) for s, n in files if (s["repo"], n) not in completed]

    def download(item):
        spec, name = item
        return (
            spec,
            name,
            hf_hub_download(spec["repo"], name, repo_type="dataset", revision=spec["revision"]),
        )

    # Keep the number of outstanding futures bounded even for thousands of files.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(pending), 32):
            for spec, name, path in pool.map(download, pending[start : start + 32]):
                with db:
                    for row in pq.read_table(path).to_pylist():
                        text = normalize(row["text"])
                        if not text:
                            continue
                        key = hashlib.sha256(text.encode()).hexdigest()
                        provenance = {
                            k: row.get(k)
                            for k in (
                                "url",
                                "document_id",
                                "collection_id",
                                "language",
                                "text_sha256",
                                "retrieved_at",
                                "extraction_version",
                                "provenance_json",
                            )
                        }
                        provenance.update(repo=spec["repo"], revision=spec["revision"], shard=name)
                        db.execute(
                            """INSERT INTO latest VALUES (?,?,?,?,?,?,?)
                            ON CONFLICT(id) DO UPDATE SET retrieved=excluded.retrieved,
                            hash=excluded.hash,parent=excluded.parent,institution=excluded.institution,
                            provenance=excluded.provenance,text=excluded.text
                            WHERE excluded.retrieved > latest.retrieved""",
                            (
                                spec["repo"] + ":" + row["document_id"],
                                row.get("retrieved_at") or "",
                                key,
                                family(row),
                                row["source_id"],
                                json.dumps(provenance),
                                zlib.compress(text.encode()),
                            ),
                        )
                    db.execute("INSERT INTO files VALUES (?,?)", (spec["repo"], name))
            if start % 320 == 0:
                commit()
                print(
                    json.dumps(
                        {
                            "stage": "downloading",
                            "completed_shards": len(completed) + min(start + 32, len(pending)),
                            "total_shards": len(files),
                        }
                    ),
                    flush=True,
                )
    # Build final table separately: only publish the manifest after closing it.
    raw_path = folder / "raw.sqlite"
    raw_path.unlink(missing_ok=True)
    out = sqlite3.connect(raw_path)
    out.executescript("""CREATE TABLE documents (
        hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
        provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0);
        CREATE INDEX document_split ON documents(source, split, conflict);""")
    tokenizer = load_frozen()
    cursor = db.execute(
        "SELECT hash,parent,provenance,text,institution FROM latest ORDER BY hash,id"
    )
    processed = 0
    while rows := cursor.fetchmany(32):
        texts = [zlib.decompress(r[3]).decode() for r in rows]
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        with out:
            for (key, parent, provenance, text, institution), ids in zip(rows, encoded):
                if any(t < FIRST_CONTENT_TOKEN_ID for t in ids.ids):
                    raise ValueError("Reserved special token found in corpus content")
                split = split_for(parent)
                previous = out.execute(
                    "SELECT split FROM documents WHERE hash=?", (key,)
                ).fetchone()
                if previous:
                    if previous[0] != split:
                        out.execute("UPDATE documents SET conflict=1 WHERE hash=?", (key,))
                    continue
                out.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                    (
                        key,
                        "eu",
                        split,
                        parent,
                        json.dumps(
                            {"institution": institution, "upstream": json.loads(provenance)}
                        ),
                        text,
                        len(ids.ids) + 1,
                    ),
                )
        processed += len(rows)
        if processed % 1024 == 0:
            print(
                json.dumps({"stage": "counting", "selected_rows_processed": processed}), flush=True
            )
    counts = {
        split: {"documents": n, "loss_tokens": tokens}
        for split, n, tokens in out.execute(
            "SELECT split,count(*),sum(loss_tokens) FROM documents WHERE conflict=0 GROUP BY split"
        )
    }
    institutions = {
        name: {"documents": n, "loss_tokens": tokens}
        for name, n, tokens in out.execute(
            "SELECT json_extract(provenance,'$.institution'),count(*),sum(loss_tokens) "
            "FROM documents WHERE conflict=0 AND split='train' GROUP BY 1"
        )
    }
    excluded = out.execute("SELECT count(*) FROM documents WHERE conflict=1").fetchone()[0]
    latest_rows = db.execute("SELECT count(*) FROM latest").fetchone()[0]
    out.close()
    db.close()
    if any(counts.get(s, {}).get("loss_tokens", 0) < 1 for s in ("train", "validation")):
        raise ValueError("Snapshot needs nonempty training and validation splits")
    manifest = {
        "format": "scholarly-raw-v1",
        "training_ready": False,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(raw_path)},
        "sources": {"eu": {"weight_percent": 100, "splits": counts}},
        "continuation_passes": 1,
        "institution_train": institutions,
        "latest_rows": latest_rows,
        "cross_split_exact_duplicates_excluded": excluded,
        "preparation": identity,
        "pending": [
            "global near deduplication",
            "cross-language parent reconciliation beyond explicit source identities",
            "source eligibility and contamination review",
        ],
    }
    temporary = manifest_path.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    commit()
    return manifest


def migrate_checkpoint(parent_path, output, manifest_path, config):
    """Make a strict-resume seed for a new dataset phase; never edit the parent."""
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.raw_token_stream import RawTokenMixture

    from trainer.distributed_train import source_hashes

    parent_path, output = Path(parent_path), Path(output)
    if output.exists():
        raise ValueError("Refuse seed checkpoint overwrite")
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    old = saved["config"]
    for field in ("n_layer", "n_head", "n_embd", "context", "world_size"):
        if old[field] != getattr(config, field):
            raise ValueError(f"Parent architecture/GPU mismatch: {field}")
    if (
        saved.get("format") != "scholarly-ddp-v1"
        or saved["data_identity"]["tokenizer_sha256"] != TOKENIZER_SHA256
    ):
        raise ValueError("Unexpected parent checkpoint or tokenizer")
    if saved["torch_version"] != str(torch.__version__):
        raise ValueError("Migration requires the parent's PyTorch version")
    model = GPT(
        ModelConfig(
            vocab_size=32768,
            block_size=config.context,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )
    )
    model.load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite parent model")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.peak_lr, betas=(0.9, 0.95), weight_decay=config.weight_decay
    )
    optimizer.load_state_dict(saved["optimizer"])
    if len(saved["rng"]) != config.world_size or not optimizer.state:
        raise ValueError("Missing parent RNG/optimizer state")
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError("Invalid parent Adam moments")
    data = RawTokenMixture(manifest_path, context=config.context, allow_unreviewed=True)
    if config.target_tokens != data.sources["eu"]["splits"]["train"]["loss_tokens"]:
        raise ValueError("Continuation must target exactly one pass of the prepared corpus")
    lineage = {
        "parent_checkpoint": str(parent_path),
        "parent_sha256": digest(parent_path),
        "parent_tokens": saved["tokens"],
        "parent_step": saved["step"],
        "parent_source_sha256": saved["source_sha256"],
        "optimizer": "preserved Adam moments and optimizer update counts",
        "schedule": "new phase, constant parent terminal LR; new data cursor",
        "phase_tokens_start": 0,
    }
    payload = {
        **saved,
        "config": config.as_dict(),
        "source_sha256": source_hashes(),
        "data_identity": data.identity,
        "data": data.state_dict(),
        "step": 0,
        "tokens": 0,
        "lineage": lineage,
    }
    data.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    return lineage
