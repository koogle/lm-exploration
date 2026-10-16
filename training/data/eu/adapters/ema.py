"""EMA inventory adapters: enumerate document links, never train on JSON summaries."""

import hashlib
import json
import re
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

BASE = "https://www.ema.europa.eu/en/documents/report/"
NEWS = BASE + "news-json-report_en.json"
NON_EPAR = BASE + "documents-output-non_epar_documents_json-report_en.json"
EPAR = BASE + "documents-output-epar_documents_json-report_en.json"
COMMITTEE_TYPES = {"agenda", "minutes", "committee-report"}
GUIDELINE_TYPES = {"scientific-guideline"}
COLLECTIONS = {
    "news": {
        "name": "EMA news and press releases",
        "description": "Native HTML news, including committee highlights and speeches.",
        "formats": ["html"],
        "inventory_url": NEWS,
    },
    "scientific_guidelines": {
        "name": "EMA scientific guideline documents",
        "description": "Full scientific-guideline attachments from the non-EPAR inventory.",
        "formats": ["pdf", "docx", "doc"],
        "inventory_url": NON_EPAR,
    },
    "committee_documents": {
        "name": "EMA agendas, minutes and committee reports",
        "description": "Inventory types agenda, minutes and committee-report; includes workshops.",
        "formats": ["pdf", "docx", "doc"],
        "inventory_url": NON_EPAR,
    },
    "non_epar_documents": {
        "name": "EMA other non-EPAR documents",
        "description": "Non-EPAR documents excluding separately partitioned guidelines/committee types.",
        "formats": ["pdf", "docx", "doc", "html", "xml", "txt"],
        "inventory_url": NON_EPAR,
    },
    "epar_documents": {
        "name": "EMA European public assessment report documents",
        "description": "EPAR assessments, product information, overviews and associated documents.",
        "formats": ["pdf", "docx", "doc"],
        "inventory_url": EPAR,
    },
}


def _date(value):
    if not value:
        return None
    value = str(value).strip()
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        return datetime.strptime(value, "%d/%m/%Y").date().isoformat()
    return value


def _url(value):
    if not isinstance(value, str):
        return None
    parts = urlsplit(value.strip())
    if parts.scheme != "https" or parts.hostname != "www.ema.europa.eu":
        return None
    if not re.match(r"^/[a-z]{2}/(?:news|documents)/", parts.path):
        return None
    return urlunsplit(parts._replace(fragment=""))


def _included(collection_id, item):
    kind = item.get("type", "")
    if collection_id == "scientific_guidelines":
        return kind in GUIDELINE_TYPES
    if collection_id == "committee_documents":
        return kind in COMMITTEE_TYPES
    if collection_id == "non_epar_documents":
        return kind not in COMMITTEE_TYPES | GUIDELINE_TYPES
    return True


def enumerate_collection(collection_id, fetch, limit=1000):
    """Fetch one complete inventory; return a deterministic bounded language-level manifest.

    fetch(url) must enforce network/size bounds and return (body_bytes, provenance).
    Cache fetches across collections to avoid downloading NON_EPAR more than once.
    complete describes this inventory subset, not institution-wide historical coverage.
    """
    if collection_id not in COLLECTIONS:
        raise ValueError(f"Unknown EMA collection: {collection_id}")
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    inventory_url = COLLECTIONS[collection_id]["inventory_url"]
    body, provenance = fetch(inventory_url)
    if len(body) > 64 * 1024 * 1024:
        raise ValueError("EMA inventory exceeds 64 MiB bound")
    payload = json.loads(body)
    items = payload.get("data")
    if not isinstance(items, list):
        raise TypeError("EMA inventory lacks data array")
    meta = payload.get("meta", {})
    reported_total = meta.get("total_records")
    if reported_total is not None and int(reported_total) != len(items):
        raise ValueError(f"EMA inventory count mismatch: {reported_total} vs {len(items)}")
    records = {}
    skipped = 0
    matched = 0
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("EMA inventory data item is not an object")
        if not _included(collection_id, item):
            continue
        matched += 1
        original_url = item.get("news_url") or item.get("document_url")
        variants = [(None, original_url)]
        translations = item.get("translations", {})
        if not isinstance(translations, dict):
            raise TypeError("EMA translations must be a language-to-URL mapping")
        variants.extend(sorted(translations.items()))
        for language, candidate in variants:
            url = _url(candidate)
            if not url:
                skipped += 1
                continue
            language = language or urlsplit(url).path.split("/")[1]
            native_id = str(item.get("id") or hashlib.sha256(url.encode()).hexdigest()[:24])
            suffix = urlsplit(url).path.rsplit(".", 1)[-1].lower()
            fmt = suffix if suffix in {"pdf", "txt", "xml", "doc", "docx", "html"} else "html"
            records.setdefault(
                url,
                {
                    "source_id": "ema",
                    "collection_id": collection_id,
                    "document_id": f"ema:{native_id}:{language}",
                    "url": url,
                    "title": item.get("title") or item.get("name") or "",
                    "language": language,
                    "document_type": item.get("type")
                    or ("press-release" if item.get("press_release") == "Yes" else "news"),
                    "format": fmt,
                    "published_at": _date(item.get("first_published_date")),
                    "modified_at": _date(item.get("last_updated_date")),
                    "inventory_document_id": item.get("id"),
                    "reference_number": item.get("reference_number"),
                    "inventory_url": inventory_url,
                    "original_document_url": original_url,
                },
            )
    ordered = sorted(
        records.values(),
        key=lambda r: (r["modified_at"] or r["published_at"] or "", r["document_id"], r["url"]),
        reverse=True,
    )
    selected = ordered[:limit]
    limitations = [
        "Coverage is the published inventory, not all EMA websites or historical deletions.",
        "Newest modified/published records first; bounded pilot is not a representative sample.",
        "Inventory metadata is not full text; download and extract each document separately.",
    ]
    if len(ordered) > limit:
        limitations.append(
            f"Pilot limit {limit} leaves {len(ordered) - limit} candidates unselected."
        )
    if skipped:
        limitations.append(f"Skipped {skipped} missing or unsupported document URLs.")
    return {
        "records": selected,
        "enumeration": {
            "strategy": "complete JSON inventory; type partition; expand translations; latest first",
            "inventory_urls": [inventory_url],
            "inventory_provenance": provenance,
            "inventory_timestamp": meta.get("timestamp"),
            "reported_total": reported_total,
            "inventory_records": len(items),
            "matching_inventory_records": matched,
            "available_document_language_records": len(ordered),
            "enumerated_count": len(selected),
            "skipped_urls": skipped,
            "complete": len(selected) == len(ordered) and skipped == 0,
            "limitations": limitations,
        },
    }
