# EMA exact-token yield and scheduling — 18 October 2026

Pinned source and control revisions are in the companion JSON. Counts use only
cached `o200k_base` entries matching current shard hash, path and row count.
Tokens per completed candidate include duplicates in the denominator; uncounted
new completions make these values conservative. These are historical averages,
not forecasts for future candidates.

| Collection | Completed candidates | Audited rows | Verified tokens | Tokens / completed candidate |
| --- | ---: | ---: | ---: | ---: |
| committee_documents | 574 | 570 | 5,375,008 | 9,364.1 |
| epar_assessment_reports | 1,888 | 1,726 | 96,781,806 | 51,261.5 |
| epar_overviews | 645 | 641 | 1,511,894 | 2,344.0 |
| epar_product_information | 2,606 | 2,586 | 115,951,856 | 44,494.2 |
| epar_scientific_discussion_variations | 191 | 181 | 2,362,456 | 12,368.9 |
| epar_scientific_discussions | 442 | 330 | 7,568,660 | 17,123.7 |
| epar_variation_reports | 423 | 363 | 11,053,406 | 26,131.0 |
| news | 625 | 617 | 678,169 | 1,085.1 |
| non_epar_documents | 546 | 538 | 2,725,107 | 4,991.0 |
| scientific_guidelines | 623 | 618 | 6,166,238 | 9,897.7 |

Assessment reports have 1,888 completed candidates from 1,891 catalog entries;
the three remaining entries are oversized-file failures, not a fresh high-yield
queue. Product information has 2,606 completions from 55,216 candidates, with
one completed exact duplicate. Its measured yield is substantially higher than
remaining variation reports, discussions, overviews, news or non-EPAR documents.

Both mappings therefore raise product-information weight from 6 to 20, within
the existing scheduler maximum. Other weights remain unchanged, preserving a
coverage floor for every nonempty collection. The weighted scheduler skips empty
queues. This changes which eligible document is requested next, not the ten-second
EMA pacing, concurrency, per-round attempt/time limits, document cap, campaign
cap, audit-pair cap or source reservations. No duplicate writer is launched.

Multilingual product information intentionally retains distinct translated text
and related-version provenance. Exact text deduplication continues. Historical
yield and language composition can change as collection advances; this decision
must be revisited using later pinned audit results. It does not predict when the
billion-token target will be met or authorize additional operational budget.

Validation: 279 tests passed, plus Ruff lint/format and whitespace checks.
Deployment receipts confirmed the existing collection and manual apps and the
recovery app with its schedule disabled. Current worker images may retain the
old weights until subsequent rounds; existing source call trees remain running.
