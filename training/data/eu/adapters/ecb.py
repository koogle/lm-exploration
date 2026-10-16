"""Enumerate ECB's public, versioned FOEDB publication inventory.

The site's own JavaScript uses this database. Inventories are metadata, never
training text. The caller owns robots, request pacing, retry and raw persistence.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

BASE = "https://www.ecb.europa.eu"
DATABASE = f"{BASE}/foedb/dbs/foedb/publications.en/"
COLLECTIONS = {
    "speeches": {"name": "ECB speeches", "type_ids": ["19"]},
    "press_releases": {"name": "ECB press releases", "type_ids": ["1", "92"]},
    "interviews": {"name": "ECB interviews", "type_ids": ["27"]},
    "meeting_accounts": {"name": "ECB monetary policy meeting accounts", "type_ids": ["20"]},
    "decisions": {"name": "ECB other Governing Council decisions", "type_ids": ["55"]},
    "working_papers": {"name": "ECB Working Paper Series", "type_ids": ["2"]},
    "occasional_papers": {"name": "ECB Occasional Paper Series", "type_ids": ["3"]},
    "economic_bulletin_articles": {"name": "ECB Economic Bulletin articles", "type_ids": ["24"]},
    "economic_bulletin_boxes": {"name": "ECB Economic Bulletin boxes", "type_ids": ["25"]},
}
for _key, _collection in COLLECTIONS.items():
    _collection.update(
        description=f"{_collection['name']} from the English FOEDB publication inventory; "
        "includes explicitly listed language variants and prefers HTML per language.",
        formats=["html", "pdf"],
        document_type=_key,
    )


def canonical_url(value):
    parsed = urlsplit(urljoin(BASE + "/", value))
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "www.ecb.europa.eu":
        return None
    return urlunsplit(("https", "www.ecb.europa.eu", re.sub(r"/+", "/", parsed.path), "", ""))


def _representations(row):
    """Prefer native HTML over same-language PDF copies; preserve translations."""
    choices = {}
    for value in row.get("documentTypes") or []:
        if not isinstance(value, str):
            continue
        url = canonical_url(value)
        if not url:
            continue
        match = re.search(r"\.([a-z]{2,3})\.(html|pdf)$", url)
        if "_content." in url:
            continue
        if match:
            language, fmt = match.groups()
            key = language
        elif url.endswith(".pdf"):
            # Verified historical papers include ecbwp001.pdf and mb199901en.pdf.
            # The English inventory is UI metadata, not evidence of text language.
            language, fmt, key = "und", "pdf", url
        else:
            continue
        # Only explicitly same-language alternatives can safely replace each other.
        if key not in choices or fmt == "html":
            choices[key] = (url, language, fmt)
    return [value for _, value in sorted(choices.items())]


def enumerate_collection(collection_id, fetch, limit=1000):
    """Return a bounded, deterministic newest-first slice of a pinned inventory.

    ``complete`` means all eligible URLs of this type in this database snapshot,
    not all ECB publications or languages. Retrieval failures propagate so the
    harness cannot mistake an interrupted enumeration for an empty collection.
    """
    config = COLLECTIONS[collection_id]
    if limit < 1:
        raise ValueError("limit must be positive")
    inventory_urls = []

    def get(url):
        body, _ = fetch(url)
        inventory_urls.append(url)
        return json.loads(body)

    version = get(DATABASE + "versions.json")[0]
    snapshot = DATABASE + str(version["version"]) + "/" + version["hash"] + "/"
    metadata = get(snapshot + "metadata.json")
    header = metadata["header"]
    required = {"id", "pub_timestamp", "documentTypes", "publicationProperties", "type"}
    if not required.issubset(header):
        raise ValueError("ECB FOEDB schema changed: missing required columns")
    types = get(snapshot + "indexes/type/index.json")
    positions = set()
    reported_total = 0
    for index_id, entry in enumerate(types):
        if str(entry["value"]) not in config["type_ids"]:
            continue
        reported_total += entry["total_records"]
        for chunk in range(len(entry["first_sort_id_per_chunk"])):
            positions.update(get(snapshot + f"indexes/type/{index_id}/chunk_{chunk}.json"))
    if not reported_total:
        raise ValueError(f"ECB inventory has no expected type for {collection_id}")
    records, seen, chunks = [], set(), {}
    exhausted = True
    scanned = 0
    for position in sorted(positions):
        chunk_id = position // metadata["chunk_size"]
        if chunk_id not in chunks:
            group = chunk_id // metadata["chunk_group_size"]
            chunks[chunk_id] = get(snapshot + f"data/{group}/chunk_{chunk_id}.json")
        offset = (position % metadata["chunk_size"]) * len(header)
        values = chunks[chunk_id][offset : offset + len(header)]
        if len(values) != len(header):
            raise ValueError("ECB inventory chunk is truncated")
        row = dict(zip(header, values, strict=True))
        if str(row["type"]) not in config["type_ids"]:
            raise ValueError("ECB index and data types disagree")
        scanned += 1
        for url, language, fmt in _representations(row):
            if url in seen:
                continue
            if len(records) >= limit:
                exhausted = False
                break
            seen.add(url)
            props = row.get("publicationProperties") or {}
            records.append(
                {
                    "source_id": "ecb",
                    "collection_id": collection_id,
                    "document_id": hashlib.sha256(url.encode()).hexdigest(),
                    "source_document_id": str(row["id"]),
                    "url": url,
                    "title": props.get("Title", "").strip(),
                    "language": language,
                    "document_type": collection_id,
                    "format": fmt,
                    "published_at": datetime.fromtimestamp(row["pub_timestamp"], UTC).isoformat(),
                    "modified_at": None,
                }
            )
        if not exhausted:
            break
    limitations = [
        (
            "English publication database only; explicitly listed translations retained. "
            "Other language databases and linked attachments are not enumerated."
        ),
        "Reported total counts publication records, not language/format URLs.",
        "HTML preferred over PDF for each publication-language pair.",
        "Historical PDFs without explicit language suffix are retained with language=und.",
        "Inventory coverage is not a claim of complete institutional or historical coverage.",
    ]
    if not exhausted:
        limitations.append(
            f"Output limited to {limit} URLs; rerun with a larger limit to backfill."
        )
    return {
        "records": records,
        "enumeration": {
            "strategy": "ECB FOEDB versioned type indexes and publication chunks",
            "inventory_urls": inventory_urls,
            "snapshot_version": version["version"],
            "snapshot_hash": version["hash"],
            "reported_total": reported_total,
            "database_total_records": metadata["total_records"],
            "publication_records_scanned": scanned,
            "enumerated_count": len(records),
            "complete": exhausted,
            "limitations": limitations,
        },
    }
