# Commission Press Corner: enumerable archive and full text

Observed 2026-10-17 UTC, 04:30–04:33. Official HTTP probes used the inherited proxy and CA trust, checked robots, and waited at least 2.1 seconds between requests to ec.europa.eu. No excluded path was requested. Evidence bodies and request/status logs are in ignored `artifacts/commission-scale-20261017/`.

## What now works

The official Angular frontend bundles expose these public interfaces:

- Inventory: `https://ec.europa.eu/commission/presscorner/api/search?language=en&pagesize=100&pagenumber=1`
- Full document: `https://ec.europa.eu/commission/presscorner/api/documents?reference=IP%2F26%2F2175&language=en&etrans=false`

Both returned HTTP 200 JSON. Full-document text is `docuLanguageResource.htmlContent`, with official quotes separately in `dolaQuoteResources`. The adapter preserves quotes and author attribution and excludes unrelated contacts and search summaries. It does not rewrite or summarize text.

Search reports **131,142 records**, consistently across English, French and German preference queries. This is a document-family search total that includes fallback-language results, not 131,142 native records in each language. Do not multiply it by 24.

Requests use **1-based `pagenumber`**, while responses report **0-based `pageNumber`**. Consecutive ten-item pages were disjoint. The 100-item page size was verified. French page 50 at size 100 reached October 2024; 49 of its 100 rows were explicitly French and not machine-translated. The final ten-item English-preference page, number 13,115, contained two **French** documents from 1975. One was retrieved successfully as native full text. Thus the public archive is materially deeper than the RSS window.

## Language and silent-fallback traps

`language=fr` is a preference, not an exact language filter. Current search pages mix English fallback records, French eTranslations, and genuine French documents. The same is possible for other languages.

`etrans=false` on a French detail request for a machine-translated release silently returned the English original, HTTP 200. The adapter rejects this mismatch instead of labeling English as French. It also requires explicit `isETranslated == false` in search and `etranslated == false` in the full document. Human-produced official translations are retained; “native” here means official text not labeled eTranslation, not necessarily the original authoring language. Translation-family IDs and available-language metadata are retained.

Underscore references in the JSON endpoint, such as `IP_26_2175`, returned empty HTTP 200; use slash references such as `IP/26/2175` URL-encoded as `%2F`. Detail-page paths use underscores, which must not be confused with API reference syntax.

## Measured extraction proof

The adapter transformation followed by the shared HTML parser produced:

| Document | Validated language | Characters | Whitespace words |
| --- | --- | ---: | ---: |
| IP/26/2175 | en | 4,567 | 656 |
| QANDA/24/5101 | fr | 12,279 | 1,767 |
| DOC/75/1 | fr | 9,882 | 1,478 |

These are purposive examples proving current, translated-official, and historical extraction, not a representative size sample. No total-token forecast follows from three examples. A local attempt to load cl100k_base failed because its tokenizer download endpoint returned proxy HTTP 403; these figures remain words/characters, not token measurements.

## Runnable adapter and explicit bounds

`adapters/commission_press.py` implements the shared `COLLECTIONS`, `enumerate_collection(collection_id, fetch, limit)`, and `extract_document(candidate, data)` interfaces.

Initial collection IDs: `releases_en`, `releases_fr`, `releases_de`. Each scans at most **50 pages × 100 search records = 5,000 rows** and stops earlier once its output limit is reached. Per-language filtering may yield fewer candidates than the scan limit. These limits bound startup; they do not enumerate the entire archive. Pagination errors and repeated pages raise errors; all runs conservatively report incomplete coverage because live offset pagination is not an immutable snapshot. New insertions can move page boundaries. Add date-window enumeration or a persistent frontier before attempting complete historical upkeep.

`extract_document` returns `(html_bytes, metadata)` and rejects reference/language mismatch, absent full bodies, and missing/true translation flags. Shared pipeline integration must use transformed bytes for parsing while retaining original JSON bytes, raw hash, source URL, and transform metadata for provenance. No generic extractor should parse the entire JSON as training text.

Date filters were attempted with ISO date, Unix milliseconds, and day/month/year strings; each returned HTTP 400. Their accepted format remains unresolved. Do not silently claim snapshot stability from unverified date filtering.

Nine focused fixture tests pass with `uv run python -m pytest -q tests/test_commission_press_adapter.py`; Ruff checks pass. Tests cover pagination, repeated pages, language fallback, translation filtering, reference validation, full-body-only extraction, and zero-fetch bounds. This agent launched no Modal jobs and made no Hugging Face mutations; parent integration owns deployment.
