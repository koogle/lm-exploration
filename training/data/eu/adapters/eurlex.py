"""CELLAR legislation XHTML inventory; SPARQL metadata is never corpus text."""

import hashlib
import json
import re
from urllib.parse import urlencode

ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
CELLAR = "http://publications.europa.eu/resource/cellar/"
PAGE_SIZE = 500
COLLECTIONS = {
    f"legislation_{year}": {
        "name": f"EU legislation {year}, multilingual XHTML",
        "description": "CELEX sector 3, legal year partition; all available XHTML languages.",
        "formats": ["html"],
        "year": year,
    }
    for year in range(1952, 2027)
}
LANGUAGES = dict(
    zip(
        [
            "BUL",
            "CES",
            "DAN",
            "DEU",
            "ELL",
            "ENG",
            "EST",
            "FIN",
            "FRA",
            "GLE",
            "HRV",
            "HUN",
            "ITA",
            "LAV",
            "LIT",
            "MLT",
            "NLD",
            "POL",
            "POR",
            "RON",
            "SLK",
            "SLV",
            "SPA",
            "SWE",
        ],
        [
            "bg",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "et",
            "fi",
            "fr",
            "ga",
            "hr",
            "hu",
            "it",
            "lv",
            "lt",
            "mt",
            "nl",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
        ],
        strict=True,
    )
)
ITEM_PATTERN = re.compile(
    r"https?://publications\.europa\.eu/resource/cellar/[a-f0-9-]+\.\d+\.\d+/[^/?#]+$"
)


def _query(year, after, size):
    cursor = f"FILTER(STR(?item) > {json.dumps(after)})" if after else ""
    return f'''PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?work ?celex ?date ?e ?lang ?m ?item WHERE {{
?work cdm:resource_legal_id_sector "3"^^xsd:string;
 cdm:resource_legal_year "{year}"^^xsd:gYear;
 cdm:resource_legal_id_celex ?celex; cdm:work_date_document ?date.
?e cdm:expression_belongs_to_work ?work; cdm:expression_uses_language ?lang.
?m cdm:manifestation_manifests_expression ?e; cdm:manifestation_type "xhtml"^^xsd:string.
?item cdm:item_belongs_to_manifestation ?m.
{cursor}
}} ORDER BY STR(?item) LIMIT {size}'''


def enumerate_collection(collection_id, fetch, limit=1000):
    """Enumerate a bounded year using item-URI keyset pagination.

    fetch enforces robots, pacing and HTTP size/deadline bounds. The live graph
    is not a snapshot: save the resulting inventory for durable collection.
    """
    if collection_id not in COLLECTIONS:
        raise ValueError(f"Unknown EUR-Lex collection: {collection_id}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    records, urls, provenance = [], [], []
    seen = set()
    cursor = ""
    exhausted = False
    while len(records) < limit:
        size = min(PAGE_SIZE, limit - len(records))
        query = _query(COLLECTIONS[collection_id]["year"], cursor, size)
        url = (
            ENDPOINT
            + "?"
            + urlencode({"query": query, "format": "application/sparql-results+json"})
        )
        body, observed = fetch(url)
        payload = json.loads(body)
        rows = payload["results"]["bindings"]
        if not isinstance(rows, list) or len(rows) > size:
            raise ValueError("Unexpected CELLAR result page")
        urls.append(url)
        provenance.append(observed)
        last = cursor
        for row in rows:
            values = {
                key: row[key]["value"]
                for key in ("work", "celex", "date", "e", "lang", "m", "item")
            }
            item = values["item"]
            if not ITEM_PATTERN.fullmatch(item):
                raise ValueError("CELLAR returned unexpected content item URI")
            if item < last or item <= cursor:
                raise ValueError("CELLAR pagination did not advance in item order")
            last = item
            if item in seen:
                continue
            seen.add(item)
            language_code = values["lang"].rsplit("/", 1)[-1]
            if not values["lang"].startswith(
                "http://publications.europa.eu/resource/authority/language/"
            ):
                raise ValueError("CELLAR returned unexpected language authority")
            records.append(
                {
                    "source_id": "eurlex",
                    "collection_id": collection_id,
                    "document_id": "cellar:" + hashlib.sha256(item.encode()).hexdigest(),
                    "source_document_id": values["celex"],
                    "url": "https://" + item.split("://", 1)[1],
                    "title": "",
                    "language": LANGUAGES.get(language_code, language_code.lower()),
                    "document_type": "legislation",
                    "format": "html",
                    "published_at": values["date"],
                    "modified_at": None,
                    "celex": values["celex"],
                    "cellar_work": values["work"],
                    "cellar_expression": values["e"],
                    "cellar_manifestation": values["m"],
                    "cellar_item": item,
                    "language_authority": values["lang"],
                    "date_semantics": "work_date_document, not necessarily journal publication date",
                    "inventory_url": url,
                }
            )
        if rows and last <= cursor:
            raise ValueError("CELLAR pagination made no progress")
        cursor = last
        if len(rows) < size:
            exhausted = True
            break
    return {
        "records": records,
        "enumeration": {
            "strategy": "CELLAR SPARQL sector-3/year XHTML items, item-URI keyset pagination",
            "inventory_urls": urls,
            "inventory_provenance": provenance,
            "enumerated_count": len(records),
            "selection_limit": limit,
            "complete": exhausted,
            "last_item_cursor": cursor,
            "limitations": [
                "Only sector 3 works with legal year, document date, language and XHTML item metadata.",
                "No PDF/Formex-only works, treaties, case law or consolidated sector-0 documents.",
                "Each concrete XHTML item retained; multiple items can belong to one expression.",
                "Live graph is not snapshot-isolated; save inventory and reconcile future refreshes.",
                "CELEX lexical/item order is not representative sampling or relevance ranking.",
                "No modification timestamps: unchanged URLs require separate revalidation.",
                "Reaching limit does not establish partition exhaustion or total corpus size.",
            ],
        },
    }
