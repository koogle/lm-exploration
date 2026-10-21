# CELLAR legislation enumeration, 2026-10-17

## Result

The public CELLAR SPARQL endpoint supports **working multilingual legislation
inventory enumeration without the EUR-Lex frontend**. A bounded live test
returned **1,000 concrete XHTML item URLs, 54 CELEX works and 24 languages** from
the 2024 legal-year partition, in **8.32 seconds** (two 500-row pages, including
robots/pacing overhead). This is an inventory count, not extracted documents,
tokens, representative sampling or partition exhaustion.

Implemented [../adapters/eurlex.py](../adapters/eurlex.py), compatible with the
shared `enumerate_collection(collection_id, fetch, limit)` interface. Collections
are `legislation_1952` through `legislation_2026`. Parent integration owns source
mapping, HF repository configuration and any cloud collection launches. This
investigation made no Hugging Face writes or Modal launches.

## Live evidence

All new requests used the shared `Collector`, its robots policy, redirect checks,
30-second HTTP timeout, response-size bound and **two-second minimum spacing**.
Probes began at **2026-10-17 04:27 UTC**. Exact request URLs, UTC timestamps,
statuses and raw SHA-256 hashes are preserved in local ignored `attempts.jsonl`
and `raw/` files beneath `artifacts/source-probes/`:

| Probe directory | Result |
| --- | --- |
| `eurlex-scale-20261017` | SPARQL minimal query HTTP 200 JSON; guessed documentation path `/en/web/cellar/cellar-data/sparql-endpoint` HTTP 404 |
| `eurlex-scale-query-20261017` | Known GDPR work and expression predicates HTTP 200 JSON |
| `lex-*` | Work→expression→manifestation→item query and 2024 sector-3 query HTTP 200 JSON |
| `eurlex-adapter-20261017` | Adapter keyset pagination: three pages of 5/5/2 rows; 12 candidates. Two document downloads and text extraction verified |
| `eurlex-scale1000-20261017` | Production 500-row page size: two pages, 1,000 candidates, 24 languages, 54 distinct CELEX works |

`https://publications.europa.eu/robots.txt` redirected to
`https://op.europa.eu/robots.txt`, which returned HTTP 200 plain text. The shared
collector allowed both endpoint and document requests under this retrieved
policy. Both hosts must be present in the source allowlist. No special browser,
authentication, challenge bypass or per-document language header was needed.

Endpoint:
`https://publications.europa.eu/webapi/rdf/sparql`

GET parameters are `query=<SPARQL>` and
`format=application/sparql-results+json`. The endpoint returned the expected JSON
format with ordinary collector headers. Exact reproducible query construction
is in `_query` in the adapter. Unlike the earlier default CELEX response, this
metadata has a defined use: enumerate full-text **item** URLs, never ingest the
metadata itself as training text.

The live metadata verifies these CDM relations:

- Work: `resource_legal_id_sector`, `resource_legal_year`,
  `resource_legal_id_celex`, `work_date_document`.
- Expression: `expression_belongs_to_work`, `expression_uses_language`.
- Manifestation: `manifestation_manifests_expression`, `manifestation_type`.
- Item: `item_belongs_to_manifestation`.

The adapter selects sector **3** and manifestation type **xhtml**. It orders by
item URI and uses `FILTER(STR(?item) > previous_item)` for keyset pagination.
Malformed response schema, unexpected item URLs, out-of-order/repeated pages and
unexpected language authorities fail explicitly. Exact item identity is retained
alongside work/expression/manifestation IDs and CELEX. ISO language authorities
are mapped to two-letter codes for the 24 official languages; unknown codes are
preserved, not silently labeled English.

## Full-text verification

The following real 2024 decision was discovered by enumeration, not supplied as a
handpicked seed: **32024D2181**, Political and Security Committee Decision (CFSP)
2024/2181 of 27 August 2024.

| Language | Concrete item path under `https://publications.europa.eu/resource/cellar/` | HTTP/MIME | Bytes | Extracted characters |
| --- | --- | --- | ---: | ---: |
| Bulgarian | `00013b08-6be8-11ef-a8ba-01aa75ed71a1.0001.03/DOC_1` | 200, `application/xhtml+xml` | 13,556 | 3,432 |
| English | `00013b08-6be8-11ef-a8ba-01aa75ed71a1.0006.03/DOC_1` | 200, `application/xhtml+xml` | 11,459 | 3,294 |

The shared parser returned journal context, the decision heading, legal text and
related document structure. It emitted BeautifulSoup's XML-parsed-as-HTML warning
for XHTML; extraction nevertheless succeeded. No tokenizer was run and no
quality-filtered corpus yield is claimed. See the ignored
`eurlex-adapter-20261017-output.txt` for extraction summaries.

## Boundaries and integration

The source needs only `publications.europa.eu` and `op.europa.eu` in its allowed
hosts. Use normal HTML parsing for the concrete XHTML records; do not set a
source-global SPARQL Accept header. The adapter explicitly requests SPARQL JSON
through the format parameter and emits `format=html` document records.

The year partitions contain only works with all required metadata and native
XHTML items. PDF-only/Formex-only works, treaties, case law, consolidated texts
in sector 0 and other CELEX sectors are excluded. Multiple concrete items for
one expression are retained rather than assuming `DOC_1` is the whole work.
No total matching count query was made. Reaching a configured limit leaves
`complete=false`; a short terminal page alone marks the selected partition
exhausted. The limit is per collection, not a full-corpus completeness guarantee.

`work_date_document` is the act/document date, not necessarily the journal date
(the verified sample is dated August 27 but appeared September 6). The adapter
retains this distinction in `date_semantics`. No last-modified field is inferred.

SPARQL reads a live graph. Keyset pagination reduces offset drift but does not
provide a frozen snapshot or guarantee capturing concurrent insertions before
the cursor. Save the full enumeration through the existing pipeline before
collection; periodically reconcile refreshed inventories. A limited run resumed
with the same saved inventory does not magically advance to the rest of a year:
increase the enumeration limit and refresh, or add explicit persisted cursor
continuation before claiming a completed backfill.

For a 2–3-hour campaign this removes the previous source-discovery blocker, but
the available collection window, request pacing, document lengths and extraction
success determine yield. The small decision above is short. These measurements
do **not** substantiate a 1B-token yield in that window. Existing EUR-Lex dataset
overlap remains a separate measurement.

Validation: `uv run python -m pytest -q tests/test_eurlex_adapter.py` — **5 passed**;
Ruff check and formatting passed for the adapter and its tests. Tests cover
language preservation, keyset advancement, limit/exhaustion semantics, repeated
pages, host validation and invalid arguments. Live validation separately covered
both reduced and production page sizes and actual text retrieval.
