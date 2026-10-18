"""Enumerate ECB's public, versioned FOEDB publication inventory.

The site's own JavaScript uses this database. Inventories are metadata, never
training text. The caller owns robots, request pacing, retry and raw persistence.
"""

import hashlib
import io
import json
import re
import unicodedata
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

from pypdf import PdfReader

BASE = "https://www.ecb.europa.eu"
DATABASE = f"{BASE}/foedb/dbs/foedb/publications.en/"
COLLECTIONS = {
    "speeches": {"name": "ECB speeches", "type_ids": ["19"]},
    "press_releases": {"name": "ECB press releases", "type_ids": ["1", "92"]},
    "interviews": {"name": "ECB interviews", "type_ids": ["27"]},
    "meeting_accounts": {"name": "ECB monetary policy meeting accounts", "type_ids": ["20"]},
    "decisions": {"name": "ECB other Governing Council decisions", "type_ids": ["55"]},
    "annual_report_translations": {"name": "ECB Annual Report translations", "type_ids": ["132"]},
    "annual_reports": {"name": "ECB Annual Report", "type_ids": ["132"]},
    "convergence_reports": {"name": "ECB Convergence Report", "type_ids": ["22"]},
    "financial_stability_reviews": {"name": "ECB Financial Stability Review", "type_ids": ["137"]},
    "statistics_papers": {"name": "ECB Statistics Paper Series", "type_ids": ["5"]},
    "research_bulletins": {"name": "ECB Research Bulletin", "type_ids": ["61"]},
    "legal_working_papers": {"name": "ECB Legal Working Paper Series", "type_ids": ["4"]},
    "discussion_papers": {"name": "ECB Discussion Paper Series", "type_ids": ["377"]},
    "central_banking_proceedings": {
        "name": "ECB Forum on Central Banking - Conference proceedings",
        "type_ids": ["155"],
    },
    "legal_conference_proceedings": {"name": "Legal conference proceedings", "type_ids": ["450"]},
    "letters_to_meps": {"name": "ECB Letters to MEPs", "type_ids": ["18"]},
    "ecb_blog": {"name": "The ECB Blog", "type_ids": ["226"]},
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
    if collection_id == "annual_report_translations":
        return enumerate_report_translations(fetch, limit=limit)
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


class _AnchorLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def report_translation_links(body, seed_url):
    """Only explicit same-path, same-report/hash HTML language counterparts."""
    seed = canonical_url(seed_url)
    match = re.fullmatch(r"(.+)\.en\.html", seed or "")
    if not match:
        return []
    parser = _AnchorLinks()
    parser.feed(body.decode("utf-8", errors="replace"))
    found = {}
    for href in parser.links:
        url = canonical_url(urljoin(seed, href))
        variant = re.fullmatch(re.escape(match[1]) + r"\.([a-z]{2,3})\.html", url or "")
        if variant and variant[1] != "en":
            found[url] = variant[1]
    return sorted(found.items())


def enumerate_report_translations(fetch, limit=1000, seed_limit=100):
    """Bounded extra HTML translations linked by English annual reports.

    No URL synthesis or recursion. Completeness concerns these inspected links,
    not all report translations, and language suffixes remain publisher metadata.
    """
    if limit < 1 or seed_limit < 1:
        raise ValueError("limits must be positive")
    base = enumerate_collection("annual_reports", fetch, limit=1000)
    seeds = [r for r in base["records"] if r["language"] == "en" and r["format"] == "html"]
    seen = {r["url"] for r in base["records"]}
    records, inspected = [], []
    complete = base["enumeration"]["complete"] and len(seeds) <= seed_limit
    for seed in seeds[:seed_limit]:
        if len(records) >= limit:
            complete = False
            break
        body, provenance = fetch(seed["url"])
        if canonical_url(provenance.get("final_url", seed["url"])) != seed["url"]:
            raise ValueError("ECB report seed redirected to a different document")
        inspected.append(seed["url"])
        for url, language in report_translation_links(body, seed["url"]):
            if url in seen:
                continue
            if len(records) >= limit:
                complete = False
                break
            seen.add(url)
            records.append(
                {
                    **seed,
                    "collection_id": "annual_report_translations",
                    "document_type": "annual_report_translations",
                    "document_id": hashlib.sha256(url.encode()).hexdigest(),
                    "url": url,
                    "language": language,
                    "discovery_url": seed["url"],
                }
            )
    return {
        "records": records,
        "enumeration": {
            **base["enumeration"],
            "strategy": "FOEDB annual reports plus explicit same-document HTML language anchors",
            "seed_pages_inspected": inspected,
            "seed_page_limit": seed_limit,
            "enumerated_count": len(records),
            "complete": complete,
            "limitations": [
                "Only English HTML seeds in a bounded 1000-URL FOEDB annual report inventory.",
                "Only explicit same-path report/hash anchors; historical PDF translations omitted.",
                "English and URLs already listed in the base inventory are excluded.",
                "Language is the publisher URL label; translated body language is not validated here.",
                "Complete means bounded linked inventory only, not website or language exhaustion.",
            ],
        },
    }


# Exact standalone assets verified on 2026-10-16 and 2026-10-17. No blanket prefix rewrite:
# inventory errors in other historical files need their own verification.
# Exact PDF identities verified in research/ecb-convergence-links-20261017.json.
VERIFIED_CONVERGENCE_REPORTS = {
    "https://www.ecb.europa.eu/press/other-publications/convergence/pdf/cr2006en.pdf": {
        "url": "https://www.ecb.europa.eu/pub/pdf/conrep/cr2006en.pdf",
        "sha256": "8e545100a568ed8752c759365a6261c82e252178427e918c57eeeb5b4b99a3d7",
    },
    "https://www.ecb.europa.eu/press/other-publications/convergence/pdf/cr2000en.pdf": {
        "url": "https://www.ecb.europa.eu/pub/pdf/conrep/cr2000en.pdf",
        "sha256": "fccdf76e68a1a61505dd630d2d4f6e5e4f30309f7fcc2338e52bcc04b87fb156",
    },
    "https://www.ecb.europa.eu/press/other-publications/convergence/pdf/cr200705en.pdf": {
        "url": "https://www.ecb.europa.eu/pub/pdf/conrep/cr200705en.pdf",
        "sha256": "22076668ff99886ef0f7d15656fea2dc2bf17471a3242a4e6af53ffdcf646cdb",
    },
}


VERIFIED_HISTORICAL_ARTICLES = {
    BASE + "/press/economic-bulletin/articles/2014/pdf/art1_mb201402en_pp77-96en.pdf": BASE
    + "/pub/pdf/other/art1_mb201402en_pp77-96en.pdf",
    BASE + "/press/economic-bulletin/articles/2014/pdf/art3_mb201401en_pp83-97en.pdf": BASE
    + "/pub/pdf/other/art3_mb201401en_pp83-97en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201412.en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201412.en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art2_mb201411_pp91-105.en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art2_mb201411_pp91-105.en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art2_mb201410_pp69-82.en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art2_mb201410_pp69-82.en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201410_pp49-68.en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201410_pp49-68.en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201409.en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201409.en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art2_mb201408_pp83-97en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art2_mb201408_pp83-97en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201408_pp65-82en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201408_pp65-82en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art3_mb201407_pp99-114en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art3_mb201407_pp99-114en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art2_mb201407_pp79-97en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art2_mb201407_pp79-97en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201407_pp63-77en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201407_pp63-77en.pdf",
    "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201406en_pp101-114en.pdf": "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201406en_pp101-114en.pdf",
}


# These patterns only propose a URL. Inferred responses require title validation.
MONTHLY_ARTICLE_FILENAMES = [
    re.compile(pattern)
    for pattern in (
        (
            r"art[1-9]\d*_mb(?P<issue>\d{6})(?:en)?_pp(?P<start>\d{1,3})-"
            r"(?P<end>\d{1,3})(?:en|\.en)\.pdf"
        ),
        r"art[1-9]\d*_mb(?P<issue>\d{6})\.en\.pdf",
        r"p{1,2}(?P<start>\d{1,3})[-_](?P<end>\d{1,3})_?mb(?P<issue>\d{6})en\.pdf",
        (
            r"mb(?P<issue>\d{6})(?:en)?_(?:pp)?(?P<start>\d{1,3})-"
            r"(?P<end>\d{1,3})en\.pdf"
        ),
    )
]


def _inferred_monthly_article_url(record):
    if record.get("collection_id") != "economic_bulletin_articles":
        return None
    path = re.fullmatch(
        r"https://www\.ecb\.europa\.eu/press/economic-bulletin/articles/"
        r"(?P<year>\d{4})/pdf/(?P<name>[^/]+)",
        record["url"],
    )
    if not path or not 1999 <= int(path["year"]) <= 2014:
        return None
    for pattern in MONTHLY_ARTICLE_FILENAMES:
        match = pattern.fullmatch(path["name"])
        if not match:
            continue
        if match["issue"][:4] != path["year"] or not 1 <= int(match["issue"][4:]) <= 12:
            continue
        if "start" in match.groupdict() and not 0 < int(match["start"]) <= int(match["end"]) <= 999:
            continue
        return BASE + "/pub/pdf/other/" + path["name"]
    return None


def _inferred_bulletin_box_url(record):
    """Propose one historical asset; filename matching alone never validates it."""
    if record.get("collection_id") != "economic_bulletin_boxes":
        return None
    path = re.fullmatch(
        r"https://www\.ecb\.europa\.eu/press/economic-bulletin/focus/"
        r"(?P<year>201[2-8])/pdf/(?P<name>[^/]+)",
        record["url"],
    )
    if not path:
        return None
    match = re.fullmatch(
        r"(?P<family>mb|eb|ebbox|ecb\.ebbox)(?P<issue>\d{6})_"
        r"(?P<kind>focus|box)?(?P<box>\d{2})\.en\.pdf",
        path["name"],
    )
    if not match or match["issue"][:4] != path["year"] or int(match["box"]) < 1:
        return None
    # Older verified samples use Monthly Bulletin filenames only. Retain the
    # opening-title check: even a plausible 2011 replacement failed that check.
    if int(path["year"]) < 2014 and match["family"] != "mb":
        return None
    allowed = {"mb": {"focus"}, "eb": {"focus", "box"}, "ebbox": {""}, "ecb.ebbox": {""}}
    if (match["kind"] or "") not in allowed[match["family"]]:
        return None
    maximum = 12 if match["family"] == "mb" else 8
    if not 1 <= int(match["issue"][4:]) <= maximum:
        return None
    return BASE + "/pub/pdf/other/" + path["name"]


def alternate_document_urls(record):
    """Return one bounded candidate after primary 404; validate inferred PDF bodies.

    Keep original inventory identity and record both fetch URLs in provenance.
    Never use this function without validate_alternate before normal extraction.
    """
    if record.get("collection_id") == "convergence_reports":
        verified = VERIFIED_CONVERGENCE_REPORTS.get(record["url"])
        return [verified["url"]] if verified else []
    if record.get("collection_id") == "economic_bulletin_boxes":
        target = _inferred_bulletin_box_url(record)
        return [target] if target else []
    if record.get("collection_id") != "economic_bulletin_articles":
        return []
    target = VERIFIED_HISTORICAL_ARTICLES.get(record["url"]) or _inferred_monthly_article_url(
        record
    )
    return [target] if target else []


def _normalize_title(text):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", text).casefold())


def validate_alternate(record, data):
    """Reject inferred PDF replacements unless their opening pages contain the title.

    Three convergence replacements must match verified PDF SHA-256 hashes.
    Thirteen article mappings have recorded identity evidence; other article URLs
    must pass strict historical filename rules and conservative title checks.
    Inferred standalone boxes are restricted to 2012–2018 and at most 20 pages;
    before 2014 only Monthly Bulletin filenames qualify.
    This check does not turn PDF metadata or a full issue into an article.
    """
    if record.get("collection_id") == "convergence_reports":
        verified = VERIFIED_CONVERGENCE_REPORTS.get(record["url"])
        if not verified or hashlib.sha256(data).hexdigest() != verified["sha256"]:
            raise ValueError("ECB convergence alternate does not match verified PDF identity")
        return
    if (
        record.get("collection_id") == "economic_bulletin_articles"
        and record["url"] in VERIFIED_HISTORICAL_ARTICLES
    ):
        return
    if not (_inferred_monthly_article_url(record) or _inferred_bulletin_box_url(record)):
        raise ValueError("ECB alternate is outside the supported historical article or box rule")
    title = record.get("title") or ""
    normalized = _normalize_title(title)
    if len(normalized) < 12 or len(re.findall(r"[a-zA-Z]+", title)) < 3:
        raise ValueError("ECB alternate title is missing or too short for identity validation")
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("ECB alternate response is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if record.get("collection_id") == "economic_bulletin_boxes" and len(reader.pages) > 20:
            raise ValueError("ECB box alternate is too long for the supported standalone-box rule")
        opening = "\n".join(page.extract_text() or "" for page in reader.pages[:2])
    except Exception as error:
        raise ValueError("ECB alternate PDF opening pages could not be extracted") from error
    if normalized not in _normalize_title(opening):
        raise ValueError("ECB alternate PDF title mismatch in first two pages")
