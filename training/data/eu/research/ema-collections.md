# EMA collection enumeration

The adapter in `adapters/ema.py` consumes official complete JSON inventories and
emits document-language records. The shared fetcher owns robots, request spacing,
body limits, raw storage and provenance. Cache inventories by URL so three
non-EPAR partitions download the 34.9 MB inventory once per refresh.

Three fresh bounded GETs decoded successfully (full bodies, not snippets):

| Inventory | Bytes | Reported and decoded records |
| --- | ---: | ---: |
| Scientific-guideline landing pages | 412,419 | 1,000 |
| Non-EPAR documents | 34,911,923 | 50,525 |
| EPAR documents | 29,222,731 | 20,266 |

The landing-page guideline inventory is not full guideline text. The adapter uses
non-EPAR `type=scientific-guideline` attachments instead. Native HTML is available
for news; guidelines and committee documents predominantly expose PDFs. No OCR
or document rewriting is introduced. Additional file formats remain candidates
for the shared extractor, which should report unsupported formats explicitly.

| Collection ID | Inventory filter | Source records | Unique language URLs |
| --- | --- | ---: | ---: |
| `news` | News inventory | 3,894 | 3,894 |
| `scientific_guidelines` | `scientific-guideline` | 2,745 | 2,793 |
| `committee_documents` | `agenda`, `minutes`, `committee-report` | 4,926 | 5,177 |
| `non_epar_documents` | Remaining non-EPAR types | 42,854 | 106,149 |
| `epar_documents` | EPAR inventory | 20,266 | 187,959 |

News counts reuse the earlier complete saved news inventory. Committee documents
include workshop agendas: this partition reflects published document types, not
an assertion that every document originated in a formal committee. EPAR has 96
additional `/media/` landing links excluded explicitly pending investigation.
Totals include translations, which are separate language records, not independent
works. Counts do not represent downloaded full text or successful extraction.

## Durable enumeration process

1. Fetch and preserve the inventory and retrieval provenance. Validate declared
   record count against the fully decoded array; fail on truncation/mismatch.
2. Partition non-EPAR records by mutually exclusive types as above.
3. Expand `translations` language-to-URL mappings; preserve the original URL,
   institution ID, reference number, publication and modification dates.
4. Assign IDs `ema:<inventory-id>:<language>`. News lacks inventory IDs, so use a
   deterministic URL digest with language. Deduplicate exact URLs in each manifest.
5. Sort newest modification/publication first with stable ties. The adapter limit
   is explicit and `complete=false` whenever candidates remain or URLs were skipped.
6. The harness stores manifests/progress durably, selects pending records, downloads
   bodies, extracts text and checkpoints Parquet. To extend beyond a bounded
   manifest, increase the enumeration limit and reuse durable document IDs.

Recommended first run: `news`, `scientific_guidelines`, `committee_documents`,
sharing the user's 1,000-document EMA quota across collections. This recent-first
sample is not representative for token extrapolation. Native IDs track replacement
versions; the parent harness must compare modified dates/content hashes on refresh.

There were no new full-document downloads in this adapter investigation. The
existing pilot verified EMA HTML news extraction; PDF extraction success remains
a run-time measurement. Four adapter tests cover partitions/translations, stable
IDs, latest-first bounds, duplicate/unsafe URLs and inventory count validation.
Exact inventory observations and enumeration summaries are in
[ema-collections.json](ema-collections.json). Raw bodies remain ignored artifacts.
