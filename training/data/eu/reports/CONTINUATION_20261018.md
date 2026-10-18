# ECB/EMA continuation — 18 October 2026, 05:22 UTC

The goal remains active and incomplete. Pinned source checkpoints contain
**18,206 documents** and **at least 412,205,746 verified `o200k_base` tokens**
(41.22% of one billion). Neither website is demonstrably exhausted.
[Machine-readable evidence](../research/continuation-checkpoint-20261018.json)
records source/control revisions and audit coverage. Cache contributions were
matched to current shard hash, path and row count using the controller's helper.

| Source | Stored documents | Verified tokens | Audited documents | Partitioned raw objects |
| --- | ---: | ---: | ---: | ---: |
| ECB | 10,223 | 172,913,672 | 10,010 | 226 |
| EMA | 7,983 | 239,292,074 | 7,848 | 0 |

All five existing repositories were confirmed private. An ECB partitioned gzip
was downloaded at its pinned revision and its decompressed SHA-256 matched its
filename. EMA is still in the third round of its older campaign; successful
partitioned EMA writes remain to be confirmed after its successor starts. Do not
stop or duplicate that active writer merely to replace its image.

Both selected campaign call trees were inspected and remained running. The
latest reserved audit pair had completed on Modal; durable schedule state still
awaited the next hourly reconciliation. No extra audit pair was dispatched.
There are 23 campaign records of the 150 limit, 6,000 reserved document attempts,
and 14 of 48 audit pairs reserved. Caps, pacing and schedules remain unchanged.
The audit envelope needs another review before its remaining 34 pairs run out;
exhausting it would be an operational blocker, not website exhaustion.

## Repairs and validation

[Older ECB box recovery](../research/ecb-older-box-recovery-20261018.md) adds
2012–2013 Monthly Bulletin filenames to the guarded alternate-path rule. Two
live PDFs passed full opening-title validation; a 2011 sample failed and remains
excluded. The pinned inventory contains 175 qualifying candidates. They are not
counted as newly collected documents. Existing completion, cooldown and
attempted-alternate history are preserved.

The Modal CLI wrapper now uses the SDK's official entrypoint after applying the
same managed CA configuration. Direct invocation of the Click group suppressed
SDK deployment/app-list output. The updated wrapper produced explicit deployment
receipts and a readable app inventory.

Validation: 260 tests passed, Ruff lint and formatting passed, and whitespace
checks passed. Deployment receipts confirmed updates to the existing scheduled
collection app, manual collector app and recovery app. The recovery deployment
used `EU_GOAL_SCHEDULE=0`. No new app, independent schedule or source writer was
created. The token-audit app needed no code change and was left running.

## Next checks

Refresh both source checkpoints and matched token caches; inspect reservations
before any dispatch. Confirm that subsequent EMA workers save `raw-v2` objects
and that subsequent ECB workers use the older-box recovery. Keep the current
one-billion-token endpoint and operational bounds. Unsupported spreadsheet/XML
misclassification failures, oversized documents, historical links and incomplete
catalog coverage remain gaps; none establish institutional exhaustion.

## Follow-up at 05:39 UTC

[Fresh pinned status](../research/series-checkpoint-20261018.json) now records
18,414 documents (10,357 ECB; 8,057 EMA). The exact matched-shard lower bound is
still 412,205,746 tokens: the hourly audit has not yet covered 556 newer rows.
The read-only `status.py both --mapping goal_pipeline_sources.json --cached-tokens`
command now reports that distinction directly, together with durable reservations.
It does not launch audits or download corpus Parquet.

EMA's terminal campaign tree was inspected and reconciled through two calls to
the existing controller. It allocated exactly one successor:
`goal-ema-1792301652-8336f3de89`, supervisor `fc-01M56R1QMKAM6PDEJKE3GH0VY2`.
Both selected source call trees remained running at the subsequent check. There
are now 24 campaign records, 6,000 reserved attempts and 14 reserved audit pairs.
No caps or schedule intervals changed.

The successor has written 16 partitioned originals. One was downloaded at pinned
EMA revision `1ca963cf775762a4179710a01e252f5eb8b0d7f6`; decompressed SHA-256
matched its filename. [Verification evidence](../research/ema-raw-v2-confirmation-20261018.json)
closes the previously outstanding EMA raw-v2 propagation check.

Six additional exact ECB types passed live full-text sample extraction and were
added to both mappings: legal papers, discussion papers, central-banking and legal
conference proceedings, MEP letters and blog posts. Their 1,057 catalog records
are not yet counted as uploaded documents. [Evidence and limitations](../research/ecb-distinct-series-20261018.md)
include passage overlap in proceedings and incomplete historical validation.
All 269 tests and lint/format/whitespace checks passed. Existing collection,
manual and schedule-disabled recovery apps were deployed successfully. Commit
`d1dab5d4ad55e2f364c06c0626811411fab812bd` was pushed to PR #14 and its private
HF code archive was downloaded and checksum-verified.

Next: check newly enumerated series on subsequent ECB workers and refresh audit
coverage after its next hourly pass. Preserve the remaining 34-pair audit and
126-campaign envelopes; review them before exhaustion. Collection continues,
and neither completion criterion has been met.
