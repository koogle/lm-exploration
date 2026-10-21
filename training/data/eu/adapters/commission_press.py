"""Official Press Corner search and full-document JSON, never leadText summaries.

The caller supplies rate-limited, robots-aware fetch and stores original bytes.
Search language is a preference, not a filter: native language and translation
flags must be checked on both inventory rows and downloaded documents.
"""

import html
import json
import re
from urllib.parse import urlencode

BASE = "https://ec.europa.eu/commission/presscorner/"
PAGE_SIZE = 100
MAX_PAGES = 50
LANGUAGES = ("en", "fr", "de")
COLLECTIONS = {
    f"releases_{language}": {
        "name": f"Commission Press Corner official {language.upper()} documents",
        "description": "Press releases, speeches, statements, Q&A and daily news; "
        "strict native-language selection excluding eTranslation.",
        "language": language,
        "formats": ["html"],
        "max_inventory_pages": MAX_PAGES,
    }
    for language in LANGUAGES
}
REFERENCE = re.compile(r"[A-Z][A-Z0-9_]*/\d{2,4}/\d+")


def document_url(reference, language):
    if not REFERENCE.fullmatch(reference) or language not in LANGUAGES:
        raise ValueError("Invalid Press Corner reference or language")
    return (
        BASE
        + "api/documents?"
        + urlencode({"reference": reference, "language": language, "etrans": "false"})
    )


def enumerate_collection(collection_id, fetch, limit=1000):
    """Bounded newest-first search; complete never implies institution-wide coverage.

    At most 50 pages / 5,000 search rows per call. Search totals include fallback
    languages and machine translations and are not native-language totals.
    Pagination is 1-based in requests, 0-based in responses. The live inventory
    can shift; duplicate rows are removed and completeness stays conservative.
    """
    language = COLLECTIONS[collection_id]["language"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    records = {}
    inventory_urls = []
    provenance = []
    totals = []
    seen_pages = set()
    scanned = rejected_language = rejected_translation = 0
    exhausted = False
    for page in range(1, MAX_PAGES + 1):
        if len(records) >= limit:
            break
        url = (
            BASE
            + "api/search?"
            + urlencode({"language": language, "pagesize": PAGE_SIZE, "pagenumber": page})
        )
        body, evidence = fetch(url)
        payload = json.loads(body)
        items = payload.get("docuLanguageListResources")
        if not isinstance(items, list):
            raise TypeError("Press Corner search lacks document array")
        total = payload.get("totalNumber")
        if not isinstance(total, int) or total < 0:
            raise ValueError("Press Corner search lacks valid total")
        if payload.get("pageNumber") != page - 1 or payload.get("pageSize") != PAGE_SIZE:
            raise ValueError("Press Corner pagination mismatch")
        fingerprint = tuple(str(row.get("ky")) for row in items)
        if items and fingerprint in seen_pages:
            raise ValueError("Press Corner repeated inventory page")
        seen_pages.add(fingerprint)
        inventory_urls.append(url)
        provenance.append(evidence)
        totals.append(total)
        for row in items:
            scanned += 1
            if str(row.get("languageCode", "")).lower() != language:
                rejected_language += 1
                continue
            if row.get("isETranslated") is not False:
                rejected_translation += 1
                continue
            reference = row.get("refCode", "")
            url = document_url(reference, language)
            key = f"commission-press:{reference}:{language}"
            records.setdefault(
                key,
                {
                    "source_id": "commission-press",
                    "collection_id": collection_id,
                    "document_id": key,
                    "url": url,
                    "title": row.get("title") or "",
                    "language": language,
                    "format": "html",
                    "document_type": (row.get("docutype") or {}).get("code"),
                    "published_at": row.get("eventDate"),
                    "modified_at": None,
                    "reference_number": reference,
                    "inventory_document_id": row.get("docuky"),
                    "inventory_language_id": row.get("ky"),
                    "is_machine_translated": False,
                    "inventory_url": inventory_urls[-1],
                    "original_document_url": BASE
                    + "detail/"
                    + language
                    + "/"
                    + reference.replace("/", "_").lower(),
                },
            )
        exhausted = page * PAGE_SIZE >= total
        if exhausted:
            break
        if not items:
            raise ValueError("Press Corner empty page before reported end")
    selected = list(records.values())[:limit]
    return {
        "records": selected,
        "enumeration": {
            "strategy": "bounded live search; exact native language; exclude eTranslation",
            "inventory_urls": inventory_urls,
            "inventory_provenance": provenance,
            "reported_total": totals[0] if totals else None,
            "reported_totals_observed": sorted(set(totals)),
            "scanned_search_rows": scanned,
            "matching_inventory_records": len(records),
            "skipped_language_fallback": rejected_language,
            "skipped_machine_translation_or_unknown": rejected_translation,
            "enumerated_count": len(selected),
            "selection_limit": limit,
            "max_pages": MAX_PAGES,
            "page_size": PAGE_SIZE,
            "selection_truncated": not exhausted or len(records) > limit,
            "complete": False,
            "limitations": [
                "Search total includes language fallbacks and machine translations, not native-language count.",
                "Live offset pagination is not an immutable snapshot; new insertions may shift pages.",
                "At most 5000 newest search rows scanned per collection; historical coverage remains incomplete.",
                "Enumeration contains no training text; full JSON htmlContent must be separately fetched and validated.",
            ],
        },
    }


def extract_document(candidate, data):
    """Return (HTML bytes, metadata), preserving original JSON in caller storage.

    Reject silent language fallback, wrong reference, absent translation flags,
    and absent htmlContent. Never fall back to title/leadText/summary.
    """
    payload = json.loads(data)
    resource = payload.get("docuLanguageResource")
    if not isinstance(resource, dict):
        raise TypeError("Press Corner full-document resource missing")
    reference = payload.get("refCd")
    language = str(resource.get("language", "")).lower()
    if reference != candidate.get("reference_number"):
        raise ValueError("Press Corner reference mismatch")
    if language != candidate.get("language"):
        raise ValueError("Press Corner language fallback or mismatch")
    if resource.get("etranslated") is not False:
        raise ValueError("Press Corner machine translation or unknown translation status")
    content = resource.get("htmlContent")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Press Corner full htmlContent missing")
    title = resource.get("title") or candidate.get("title", "")
    parts = [
        f'<html lang="{html.escape(language)}"><head><title>{html.escape(title)}</title></head>',
        "<body><article>",
        f"<h1>{html.escape(title)}</h1>",
        content,
    ]
    for quote in resource.get("dolaQuoteResources") or []:
        if quote.get("extract"):
            parts.append("<blockquote>" + html.escape(quote["extract"]) + "</blockquote>")
            author = (quote.get("author") or {}).get("authorName")
            if author:
                parts.append("<p>" + html.escape(author) + "</p>")
    parts.append("</article></body></html>")
    return "\n".join(parts).encode(), {
        "transform": "commission_press_json_htmlContent_v1",
        "reference_number": reference,
        "language": language,
        "title": title,
        "document_language_id": resource.get("ky"),
        "document_family_id": payload.get("ky"),
        "is_machine_translated": False,
        "original_language": payload.get("originalLanguage"),
        "is_original_language": resource.get("original"),
        "available_languages": payload.get("languages"),
        "machine_translation_languages": payload.get("eTranslatedlanguages"),
        "published_at": payload.get("publishDate"),
    }
