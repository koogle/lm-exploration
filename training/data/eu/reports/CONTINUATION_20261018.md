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
