"""Pinned patch records; related versions may overlap, exact patch duplicates do not."""

import gzip
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

from .overnight_corpus import LICENSES, SCHEMA
from .simple_llm import SPLIT_VERSION, canonical
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, load_frozen


def patch_record(row, repo, revision, shard, index, repositories):
    text = row.get("patch")
    parent = "code_python:" + str(row.get("repository", "")).lower()
    if parent == "code_python:" or not isinstance(text, str) or not 64 <= len(text) <= 4_000_000:
        return None
    # Keep the existing permissive-license policy. Collection metadata is evidence,
    # not a claim that all histories or per-file terms have been audited.
    evidence = row.get("repository_license")
    if not evidence:
        before, after = row.get("license_before"), row.get("license_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None

        def mit(value):
            content = str(value.get("text", "")).lower()
            return "permission is hereby granted" in content and "without restriction" in content

        evidence = "MIT" if mit(before) and mit(after) else "unknown"
    if str(evidence).lower() not in LICENSES:
        return None
    prior = repositories.get(parent)
    if prior and prior != "train":
        return None
    bucket = int(hashlib.sha256(f"{SPLIT_VERSION}:{parent}".encode()).hexdigest(), 16) % 10
    split = "train" if prior == "train" or bucket > 1 else "validation" if bucket == 0 else "test"
    if split == "test":
        return None
    metadata = {
        "repo": repo,
        "revision": revision,
        "shard": shard,
        "row": index,
        "repository": row["repository"],
        "commit": row.get("sha"),
        "license": evidence,
        "license_before": row.get("license_before"),
        "license_after": row.get("license_after"),
        "language": "diff",
        "task_family": "patches",
        "representation": "git_patch",
        "related_version_overlap_allowed": True,
    }
    return text, parent, split, metadata


def prepare(folder, datasets, exclusions, commit=lambda: None):
    from huggingface_hub import HfApi, hf_hub_download

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "raw.sqlite")
    db.executescript(
        SCHEMA
        + "CREATE TABLE IF NOT EXISTS completed (repo TEXT, revision TEXT, shard TEXT, PRIMARY KEY(repo,revision,shard)); CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY,value TEXT);"
    )
    identity = canonical({"datasets": datasets, "tokenizer": TOKENIZER_SHA256})
    saved = db.execute("SELECT value FROM identity WHERE id=1").fetchone()
    if saved and saved[0] != identity:
        raise ValueError("Patch snapshot identity changed")
    db.execute("INSERT OR IGNORE INTO identity VALUES(1,?)", (identity,))
    db.commit()
    old = sqlite3.connect(f"file:{exclusions}?mode=ro", uri=True)
    repositories = dict(old.execute("SELECT parent,split FROM repositories"))
    old.close()
    tokenizer = load_frozen()
    for dataset in datasets:
        repo, revision = dataset["repo"], dataset["revision"]
        files = sorted(
            p
            for p in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
            if p.startswith("changes/") and p.endswith(".jsonl.gz")
        )
        for shard in files:
            if db.execute(
                "SELECT 1 FROM completed WHERE repo=? AND revision=? AND shard=?",
                (repo, revision, shard),
            ).fetchone():
                continue
            downloaded = hf_hub_download(
                repo, shard, repo_type="dataset", revision=revision, local_dir=folder / "download"
            )
            with gzip.open(downloaded, "rt") as source, db:
                pending = []

                def flush(pending=pending):
                    for record, encoded in zip(
                        pending,
                        tokenizer.encode_batch([p[0] for p in pending], add_special_tokens=False),
                    ):
                        text, parent, split, metadata = record
                        if not encoded.ids or any(t < FIRST_CONTENT_TOKEN_ID for t in encoded.ids):
                            continue
                        key = hashlib.sha256(text.encode()).hexdigest()
                        db.execute(
                            "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?,0)",
                            (
                                key,
                                "code",
                                split,
                                parent,
                                canonical(metadata),
                                zlib.compress(text.encode()),
                                len(encoded.ids) + 1,
                            ),
                        )
                    pending.clear()

                for index, line in enumerate(source):
                    record = patch_record(
                        json.loads(line), repo, revision, shard, index, repositories
                    )
                    if record:
                        pending.append(record)
                    if len(pending) >= 32:
                        flush()
                flush()
                db.execute("INSERT INTO completed VALUES(?,?,?)", (repo, revision, shard))
            commit()
            Path(downloaded).unlink(missing_ok=True)
    result = {
        "datasets": datasets,
        "representation": "git_patch",
        "related_version_overlap_allowed": True,
        "splits": {
            s: {"documents": n, "tokens": t}
            for s, n, t in db.execute(
                "SELECT split,COUNT(*),SUM(loss_tokens) FROM documents GROUP BY split"
            )
        },
    }
    db.close()
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    commit()
    return result
