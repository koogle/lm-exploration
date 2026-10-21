# Additional ECB series — 18 October 2026

Six distinct publication types were identified in the official `publications_types`
catalog and tested through the existing FOEDB enumerator and full-text parser.
The publication type index reports 1,057 records across these series. These are
catalog records, not new stored documents or measured tokenizer counts.

| Series | Type | Catalog records | Sample format | Extracted characters |
| --- | ---: | ---: | --- | ---: |
| ECB Legal Working Paper Series | 4 | 23 | pdf | 162,457 |
| ECB Discussion Paper Series | 377 | 32 | pdf | 158,514 |
| ECB Forum on Central Banking - Conference proceedings | 155 | 13 | pdf | 861,210 |
| Legal conference proceedings | 450 | 10 | pdf | 761,212 |
| ECB Letters to MEPs | 18 | 723 | pdf | 6,332 |
| The ECB Blog | 226 | 256 | html | 10,721 |

Each probe enumerated only the newest eligible document (`limit=1`), with
incomplete inventory coverage explicitly retained. Full-text samples include
readable legal/policy research, conference papers, an official reply to an MEP,
and a native HTML blog article. URLs, timestamps, hashes, extraction prefixes
and pinned catalog versions are retained in the companion JSON.

The shared Collector enforced robots and five-second source pacing. Requests
used the inherited proxy and TLS configuration. No corpus checkpoint was
modified and no separate writer was started by these probes.

Both source mappings now add the six exact type partitions. Existing weighted
scheduling, per-round limits, source reservations and organization-wide text
deduplication apply. No broad Other Publication umbrella or full Economic
Bulletin issue collection was added.

Conference proceedings include contributions that may also appear as speeches;
they are complete official documents, not a claim of wholly novel text. Passage
overlap removal is not implemented. Sample validation does not establish full
historical retrieval, language accuracy throughout the inventory or website
exhaustion. Future collectors must establish persisted yield.

Validation: 269 tests passed; Ruff lint/format and whitespace checks passed.
Deployment receipts confirmed updates to the existing collection, manual and
recovery apps; the recovery schedule remained disabled. No source reservations
were reset. The existing controller reconciled EMA’s terminal call tree and
allocated one successor campaign, raising campaign records from 23 to 24.

At 05:53 UTC, all six complete bounded type inventories were persisted in ECB
revision `87c3ca84de19a746c52ac707028511a83f1847ac`. Their 1,057 candidate URLs
had zero exact overlap with prior configured collections or with each other.
No rows from these six collections had been saved at that snapshot. See
[the inventory comparison](ecb-series-inventory-overlap-20261018.json).
Exact URL separation does not remove the passage-overlap caveat for proceedings.
