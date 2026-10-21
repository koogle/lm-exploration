# Targeted EMA EPAR expansion — 17 October 2026

An offline replay of the cached official inventory confirms **4,121 additional
supported PDF URL candidates** in three exact type partitions. No new EMA network
requests were made while the collector was active.

| Collection | Exact inventory type | Base records | Supported document-language URLs |
| --- | --- | ---: | ---: |
| `epar_variation_reports` | `variation-report` | 3,329 | 3,401 |
| `epar_scientific_discussions` | `scientific-discussion` | 504 | 529 |
| `epar_scientific_discussion_variations` | `scientific-discussion-variation` | 191 | 191 |

All candidates are PDFs accepted by the existing format filter. All three
partitions enumerate completely at the current 100,000-per-collection limit,
with zero skipped unsupported formats. There are 4,024 English URLs and 97 URLs
labelled as other languages by the inventory. This is language metadata, not
independent body-language verification.

## Overlap check

Exact URL overlap is **zero** against each previously configured collection:
EPAR overviews, product information, assessment reports, news, scientific
guidelines, committee documents and other non-EPAR documents. The three new
partitions also have zero pairwise URL overlap. An additional comparison against
*all* supported URLs from the complete cached non-EPAR inventory, including URLs
beyond the current 100,000-item selected slice, also found zero overlap.

These checks establish distinct URL partitions, not distinct textual content.
Reports can repeat earlier assessments or share substantial passages; retain
organization-wide normalized-text deduplication. The unrestricted
`epar_documents` partition includes these same types and should not be added
alongside the targeted series.

## Evidence and text-yield limits

The EPAR cache was published at `2026-10-16T05:49:58Z`, contains 20,266 base
records, and has SHA-256
`44ef8e0cb70bc011c2037fa4d78252d3354bdf3ebf924cd9d00b1095efb0157a`.
Replayed the existing adapter with cached EPAR/non-EPAR/news payloads only;
expanded explicit translations, applied its URL/format guards, sorted using the
production ordering, and used the current limit of 100,000. Companion JSON
records input hashes, metadata timestamps, exact counts, overlap results and
three example candidates per new partition.

The source titles identify assessment/variation reports and scientific
discussions, making them more promising for substantive prose than package
presentation listings. Examples include “Pritor-H-C-210-WS-0254 : EPAR - Assessment
Report - Variation”, “Optruma : EPAR - Scientific Discussion”, and
“PritorPlus-H-C-414-X-0042 : EPAR - Scientific Discussion - Extension”. No exact
full-text sample for these targeted types was found in the inspected existing
source probes. Their bodies have therefore **not** been extraction-validated by
this investigation; no per-document character or token forecast is justified.
The next bounded production worker can validate extraction and record failures
without introducing a new discovery mechanism.

No adapters, mappings, HF data, or schedules were modified by this research task.
The cached inventory is not proof of current website completeness, successful
future retrieval, or coverage of deleted historical documents.
