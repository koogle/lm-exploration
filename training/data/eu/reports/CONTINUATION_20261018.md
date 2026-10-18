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

## Follow-up at 05:53 UTC

Both source call trees remain active. ECB's second round persisted all six new
type inventories, totaling 1,057 candidate URLs with no exact URL overlap against
prior collections or each other; saved rows from these series were still zero at
that snapshot. Discovery is complete only for those bounded type inventories.

The latest status read at 05:52 UTC showed 18,526 stored documents and the same
412,205,746-token matched lower bound, awaiting the next hourly audit. Campaign
records remain 24 and audit-pair reservations remain 14; no extra audit was started.

Ten remaining primary-404 convergence reports have individually verified full-PDF
replacements, including EMI reports from 1996 and 1998. The existing exact URL
and SHA-256 map now covers thirteen assets. [Recovery evidence](../research/ecb-convergence-recovery-20261018.md)
records titles, dates, issuer checks, hashes and bounded retrievals. It does not
count probe text as uploaded corpus or loosen validation to a generic rewrite.
The repair passed 269 tests and lint/format/whitespace checks, and deployment
receipts confirmed all three collector apps with the recovery schedule disabled.
Verify saved output in subsequent workers; preserve active writers and reservations.

## Follow-up at 05:58 UTC

The source checkpoints contain 18,605 documents. The six added ECB types have
begun producing stored rows: their first checkpoint contained 14 documents. The
fifteenth scheduled audit pair is active; 33 pairs remain in the unchanged
48-pair envelope. Existing source reservations remain intact (24 campaigns).

Thirteen additional Monthly Bulletin box samples, one selected non-generic title
per year from 1999–2011, passed full opening-title and standalone-length checks.
The guarded fallback now permits proposals in that year range, encompassing 967
catalog candidates. The earlier 2011 title-mismatch sample still fails validation,
with a regression test preserving that behavior. This is an opportunity count,
not stored yield. [Probe evidence](../research/ecb-box-history-1999-2011-20261018.md)
records selection bias and per-response validation requirements.

All 279 tests and lint/format/whitespace checks passed. Existing collector apps
were deployed; the recovery schedule remains disabled. The active audit pair is
allowed to finish rather than consuming another reservation for a manual audit.

## Exact-token refresh at 06:03 UTC

[The pinned checkpoint](../research/audit-progress-20261018-0603.json) contains
18,663 documents and a matched-shard lower bound of **421,344,921 o200k_base
tokens** (42.13%). ECB contributes 175,182,377 tokens over 10,416 audited rows;
EMA contributes 246,162,544 over 7,973 rows. EMA's audit was still partial at this
read, and collection continued beyond both audited row counts. These numbers
come from matching immutable shard hashes, paths and rows, not character estimates.

A local read-only status process stalled for over two minutes in the hf-xet
native downloader while downloading ECB progress. Its open file descriptors and
threads identified the Xet path; no source checkpoint had been changed. Only
that local process was terminated. Retrying with the supported HTTP fallback
succeeded and produced the pinned snapshot above:

```sh
HF_HUB_DISABLE_XET=1 uv run status.py both \
  --mapping goal_pipeline_sources.json --cached-tokens
```

This retains the inherited proxy, credentials and TLS trust; it does not alter
Modal workers or their reservations. Current runtime logs contained historical
17 October rate-limit errors, not evidence of a new audit failure. The scheduled
pair was preserved and is now advancing the exact token caches. No extra pair,
source writer, budget increase or schedule change was introduced.

## Exact yield and scheduling review at 06:07 UTC

[The refreshed checkpoint](../research/goal-checkpoint-20261018-0607.json) contains
18,743 documents and **425,356,977 verified tokens** (42.54%). Both calls in audit
pair 15 were subsequently confirmed complete. Their audit coverage is still pinned
to older source revisions, so newer documents remain outside the lower bound.
Both source campaign trees remain running; no replacement writer was launched.

The six new ECB collections have saved 100 documents in total. Legal papers,
discussion papers and legal proceedings have no failures in their current type
inventories. Central-banking proceedings have 11 saved documents and two short
HTML landing-page extraction failures; those remain unresolved coverage gaps.
The newly deployed convergence and pre-2012 box rules await subsequent workers.

[Exact EMA token-yield analysis](../research/ema-exact-token-yield-20261018.md)
shows assessment reports nearly drained (1,888 completions from 1,891 candidates),
while product information has 2,606 completions from 55,216 candidates and one
completed exact duplicate. Its matching audited tokens divided by all completed
candidates give a conservative historical yield of 44,494 tokens per candidate.
This is evidence for prioritization, not a forecast or additional verified tokens.

Both mappings increase product-information weight from 6 to 20, within the
existing scheduler maximum. All other weights and all pacing, attempt/time,
concurrency, document, campaign and audit limits remain unchanged. The three
collector apps were deployed with the recovery schedule disabled. Validation:
279 tests and lint/format/whitespace checks passed. Review later pinned yield and
remaining budgets as collection proceeds; the target is still unmet.

## Adapter repair resumption and budget review

The [06:15 pinned budget review](../research/operational-budget-review-20261018.json)
records 18,862 documents and the same 425,356,977 verified-token lower bound.
There are 126 campaign records and 33 audit pairs left within the existing caps.
Closing the 574,643,023-token gap across those pairs would require an average
17,413,425 additional verified tokens per pair. This is gap arithmetic, not a
forecast. Neither budget nor audit cadence was increased.

A controller resumption edge case could preserve a saved zero runnable-candidate
count after deploying an adapter repair. Source snapshots now reuse that count
only when its recorded adapter SHA-256 matches the deployed adapter and its
collection set matches the current mapping. Otherwise the existing inventory
minus completed-items upper bound provides a scheduling hint; the bounded worker
still decides actual eligibility using support checks, retry limits and verified
alternates. No source checkpoint, completion or failure history is reset.
An unchanged adapter's measured zero remains authoritative. Active-writer and
reservation guards are unchanged. Four regression cases cover changed, missing
and matching fingerprints, fully completed inventories and read-only snapshots.
Validation: 283 tests passed; Ruff lint/format and whitespace checks passed.

The scheduled controller and disabled-schedule recovery app both deployed
successfully. The [06:19 checkpoint](../research/goal-checkpoint-20261018-0619.json)
contains 18,942 documents; verified tokens remain 425,356,977 pending the next
scheduled audit. Existing source runs continue; no extra campaign was dispatched.
