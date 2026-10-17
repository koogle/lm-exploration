# ECB research-series expansion — 2026-10-16

Added four independently named official publication series. Names are verified
against the ECB's own `publications_types` FOEDB database, not inferred only from
URL patterns. Counts below are from the previously retained publication type
index (snapshot `1792148434/maNnqDza`), so they are observations, not live guarantees.

| Collection ID | Official name | Type ID | Inventory publications |
| --- | --- | ---: | ---: |
| working_papers | ECB Working Paper Series | 2 | 3,300 |
| occasional_papers | ECB Occasional Paper Series | 3 | 400 |
| economic_bulletin_articles | ECB Economic Bulletin - Article | 24 | 503 |
| economic_bulletin_boxes | ECB Economic Bulletin - Box | 25 | 1,829 |

These add **6,032 inventory publications** before language variants, deduplication
or extraction. Working and occasional papers are substantial PDF research texts;
recent Bulletin articles and boxes have native HTML. Historic Bulletin records
include their predecessor Monthly Bulletin articles/boxes dating back to 1999.
Full Bulletin issues were not added because they would overlap the individual
articles and boxes substantially. Statistical tables and spreadsheet-only series
remain outside this expansion.

## Historical PDF correction

The current publication database's chunk 79 explicitly lists working papers
`/pub/pdf/scpwps/ecbwp001.pdf` through `ecbwp008.pdf`, without `.en.pdf` suffixes.
It also lists old Bulletin articles such as
`/press/economic-bulletin/articles/1999/pdf/p.11_16_mb199901en.pdf`.
These previously failed the language-suffix filter. They are now retained as
PDFs with language `und`: an English metadata database is not language detection.
Unknown-language files are kept separately rather than conflated as alternatives.
Known-language HTML still takes precedence over matching-language PDF copies.

The first working paper returned HTTP 200 at 2026-10-16T13:45:20Z, 455,714 bytes,
32 PDF pages. First-page extraction recovered its title, authors and May 1999
publication date. This is a verified text-bearing PDF, not a claim that all pages
or papers extract cleanly. pypdf warned that optional fontTools would improve
handling of embedded CFF fonts; the extracted first page was readable.

## Live endpoint observations

Seven bounded requests, spaced at least five seconds per origin:

- `publications_types/versions.json`: 200; version `1784901400/snZw2SNH`.
- That version's `metadata.json`: 200; 119 types, columns
  `id_publication_type`, `publication_name`.
- That version's `data/0/chunk_0.json`: 200; all four series labels verified.
- Old publication snapshot's `data/0/chunk_79.json`: **404**.
- `publications.en/versions.json`: 200; now `1792154769/4eruD2XU`.
- Current snapshot's `data/0/chunk_79.json`: 200; historical paper and Bulletin URLs.
- `/pub/pdf/scpwps/ecbwp001.pdf`: 200; readable PDF sample as above.

**Origin snapshots can disappear when the version changes.** Pinning protects
within-run consistency but does not guarantee origin retention. Persist completed
inventories to Hugging Face; if an in-progress discovery gets a 404 for a pinned
chunk, fail and restart discovery with the current version rather than treating
that chunk as empty. The existing adapter propagates that failure. Existing
persisted HF inventories remain usable for downloading stable document URLs.

Raw probe bodies are ignored under `artifacts/ecb-enumeration/`. Code and compact
findings are versioned; no new bulk document content was committed here. Five
adapter tests and lint pass after the expansion, including historical PDF forms.
No source mapping, scheduling, or deployment changes were made by this sub-agent.
