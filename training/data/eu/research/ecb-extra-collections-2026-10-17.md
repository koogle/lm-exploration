# ECB full-report expansion — 2026-10-17

Two verified collections are implemented in the existing FOEDB adapter. The
parent owns source mapping and deployment. This investigation did not launch
workers or upload documents.

| Collection | FOEDB type | Publication rows | Eligible URLs |
| --- | ---: | ---: | ---: |
| `annual_reports` | 132 | 34 | 36 |
| `convergence_reports` | 22 | 20 | 20 |

Official labels were verified in the previously retained `publications_types`
database (see `ecb-expansion.md`). Both inventories were fully enumerated from
snapshot `1792213219/2sP68GoB` on 17 October 2026. Completeness here refers only
to eligible URLs in that English FOEDB snapshot, not website coverage. The 56
URLs are candidates, not additional collected documents. Annual-report URLs
include `en` and historical `und`; convergence URLs explicitly identify English.

## Actual text samples

- Annual Report 2025: HTTP 200 native HTML, 519,476 response bytes;
  **310,376 extracted characters**.
- Annual Report 1995: HTTP 200 PDF, but extraction recovered only 129 newline
  characters. This is an image/scanned-text gap, not useful corpus text. Normal
  whitespace cleaning and the collector's minimum text threshold reject it;
  OCR remains unimplemented.
- Convergence Report June 2026: HTTP 200 native HTML,
  **559,687 extracted characters**.
- Convergence Report June 2016: HTTP 200 text-bearing PDF,
  **626,141 extracted characters**.

These are measured extraction samples, not a whole-inventory token estimate.
No full Economic Bulletin issues were added, avoiding their substantial overlap
with already configured article/box collections.

## Missing translations and deferred scope

The cached Annual Report 2025 HTML explicitly links 24 same-document language
versions (`bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `ga`, `hr`,
`hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `sv`). These
are ordinary language-picker anchors, without `hreflang` attributes. The English
FOEDB row does not expose those translations. Exact anchor URLs are retained in
the accompanying JSON; no translation body was fetched in this probe. A future
bounded language-link discovery stage could materially expand coverage, but
should validate actual document language and content rather than assuming
language-picker links guarantee translations.

Financial Stability Review, type 137, has 44 publication records. Full historical
enumeration failed on HTTP 503 for a data chunk. A reduced latest-document probe
then stopped because `robots.txt` returned HTTP 503 twice. The collector failed
closed; no bypass or alternate route was attempted. This collection is deferred
and is **not added to the adapter** until actual full-text retrieval is validated.

## Reproducibility and limits

Used the shared `Collector` with five-second per-host spacing, normal robots
checks, inherited proxy/TLS, and a 1,000-URL bound per collection. Successful raw
responses were locally cached to resume a transient chunk-48 HTTP 503 without
re-fetching already retrieved inventory chunks. Further HTTP 503s ended the
investigation. URLs, UTC timestamps, status codes, inventory metadata and sample
provenance are in the companion JSON. Raw bodies and the probe script remain
under ignored `artifacts/ecb-extra/`.

Parameterized tests verify collection type routing, native HTML preference,
retention of distinct language versions, and common source-document identity.
All 24 ECB adapter tests and targeted Ruff checks pass. No claim of production
coverage or exhaustion follows from these samples.
