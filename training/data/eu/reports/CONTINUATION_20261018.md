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

## Additional ECB reports and unresolved source gaps

The [06:27 checkpoint](../research/goal-checkpoint-20261018-0627.json) contains
19,039 stored documents (ECB 10,713; EMA 8,326). The exact current-shard lower
bound remains 425,356,977 tokens. ECB round two and EMA's newly started round two
remain assigned to their existing campaign supervisors. A separate pinned EMA
read subsequently confirmed the new round's source mapping has product-information
weight 20; document selection had not yet begun at that observation.

[Four additional exact ECB types](../research/ecb-report-series-20261018.md)
passed native full-text samples: Macroprudential Bulletin articles (type 96),
financial integration and structure reports (140), international-role-of-the-euro
reports (143), and TARGET annual reports (142). Their catalog totals sum to 150
publication records. Both mappings add these collections with weight 3. They use
the same writer, pacing, private repositories and operational limits. The scheduled,
manual and disabled-schedule recovery apps all deployed successfully. A current
worker can retain its previous mapping until a subsequent round.

Validation: 287 tests passed, including four added type/representation cases;
Ruff lint/format and whitespace checks passed. Catalog and sample counts are
not uploaded documents or new verified tokens.

[Five failed ECB retrievals were investigated](../research/ecb-unresolved-extraction-20261018.md).
Two Sintra takeaways pages still supply only title shells; their tested content
fragments return 404. Three PDF failures reproduce an embedded-font error.
Independent extraction gives readable openings for two, but garbled output for
the third, making an unconditional fallback unsuitable. All five remain explicit
production coverage gaps; retry and failure history was preserved.

EMA round two subsequently checkpointed 20 attempted/saved documents with
product-information weight 20 in its actual scheduling weights. The pinned source
revision is `3abd78f0d5a1f43d4b56457c7fa942437da36118`; heartbeat
`2026-10-18T06:28:37.309455+00:00`. This confirms worker adoption, rather than only
a deployment or mapping-file change. Verified-token yield awaits the next audit.

The [full four-series inventory comparison](../research/ecb-report-inventory-overlap-20261018.json)
found exactly 150 candidate URLs and zero exact URL overlap with the 21 prior
collections or each other. All four selected type inventories are complete for
the pinned FOEDB snapshot. This establishes inventory separation, not text novelty.

[ECB round-three adoption](../research/ecb-report-worker-adoption-20261018.json)
is confirmed at source revision `3cbee474994c00a9e4a80d9aa1f109ca904e444b`:
round three started at 06:34:01 UTC with all four new report series in its source
mapping. Its adapter SHA-256 matches committed revision `589de406d87d2ac0fb13d34abb9058116b8998b5`,
which also contains the guarded historical-box and convergence-retrieval repairs.
At this observation discovery was still in progress and no new-series saved
rows were claimed. The source has 10,795 documents. Existing supervisor
`fc-01M56NPBCA8234P7HPEH5YB2C0` owns this final round of its current campaign.

[ECB Other publication, exact FOEDB type 23](../research/ecb-other-publications-20261018.md),
adds a separate opportunity: 772 catalog publication records, including technical
manuals, agreements and proceedings. Three selected PDFs passed native extraction
with 34,804, 182,361 and 352,419 characters. Both mappings add `other_publications`
with weight 3. The updated collector, manual and disabled-schedule recovery apps
all deployed successfully. This later addition is absent from the round-three
snapshot above and awaits a subsequent worker; it does not launch another writer.
Validation: 288 tests and lint/format/whitespace checks passed. Sample counts and
catalog totals are not saved corpus rows or new verified tokens.

The [complete type-23 inventory comparison](../research/ecb-other-inventory-overlap-20261018.json)
contains 771 supported candidate URLs from 772 catalog records, with zero exact
URL overlap against either the prior 21 collections or the four new report
series. Together the five additions provide 921 distinct candidate URLs. Two
catalog records share the same `whypricestabilityen.pdf` URL; all 772 have a
supported representation. No corpus rows or token
yield are inferred from those inventory counts.

An [offline English FOEDB scope review](../research/ecb-english-catalog-scope-20261018.md)
reconstructed all 20,162 unique publication records from 81 cached data chunks.
The currently configured types cover 14,888 catalog records; 5,274 remain in
87 unconfigured types, with 5,326 additional syntactically supported URLs. This
excludes linked annual-report translations from the top-level URL count. It is
catalog discovery evidence only, not full-text or novelty evidence. Survey and
annual-account types offer future bounded investigations, while podcast landing
pages and whole-report/article containment need care. No additional types were
enabled by this offline review, and whole-site exhaustion remains unproven.

## First persisted yield from the new ECB worker

The [06:41:58 pinned source checkpoint](../research/ecb-new-series-first-yield-20261018.json)
contains **10,813 ECB documents**. The new series have saved 10 documents:
three Macroprudential Bulletin articles, three financial-integration reports,
three international-role reports and one TARGET annual report. None of these
series has a recorded extraction failure at this checkpoint.

Convergence reports have increased from 10 to 13 saved rows, confirming three
additional verified replacements were actually ingested. Seven original
convergence failures still await their pending verified alternate attempts.
Economic Bulletin boxes increased from 739 to 740. These are persisted document
counts; their additional tokenizer contribution awaits the scheduled audit.
The existing ECB supervisor remains responsible for round three.

The [saved box's Parquet provenance](../research/ecb-first-box-recovery-20261018.json)
identifies the April 2013 box “Update of the index and weight reference year to
2010 for short-term business statistics”. Its original inventory URL returned
404, and the same-named `/pub/pdf/other/mb201304_focus04.en.pdf` supplied the saved
PDF. This confirms a 2013 recovery; it does not establish saved yield for the
newer 1999–2011 proposal range.

The [06:43 checkpoint](../research/goal-checkpoint-20261018-0643.json) contains
**19,237 documents**: ECB 10,813 and EMA 8,424. The matching-cache lower bound
remains **425,356,977 verified tokens (42.54%)**, with 24 campaign records and
15 reserved audit pairs. Source repositories remain private, the existing
campaigns are active, and the hourly audit coordinator retains its reservations.
The one-billion-token target and website-exhaustion condition remain unmet.

## 07:58–08:05 continuation: live yield and article recovery

The 07:58 checkpoint contained **20,152 stored documents** and **446,538,735
verified tokens**. All 150 candidates across the four additional ECB report
series are now saved with zero recorded failures. All 20 convergence reports
are completed with zero remaining failures, confirming all ten newly mapped
replacements were ingested. The next ECB campaign automatically adopted
`other_publications`; 21 unique documents were already saved from that type.

Current ECB campaign: `goal-ecb-1792309455-2c93965844`, supervisor
`fc-01M56ZFW59E04PCZQAJ33HS7VZ`, round one running. EMA remained in round three of
`goal-ema-1792301652-8336f3de89`, supervisor `fc-01M56R1QMKAM6PDEJKE3GH0VY2`.
Both call trees were inspected and running; no duplicate source writer or manual
replacement campaign was launched. There are 25 campaign records and 17 reserved
audit pairs. Pair 17 is `71185d0a3d39474fa7afe16a559ed1f6`, with ECB call
`fc-01M5705N60YWKGWB7ERB1JWAT2` and EMA call `fc-01M5705NKXEXME0VRGA6X60FP9`.
Its unresolved claims were preserved while the caches advanced.

The [08:05 pinned checkpoint](../research/goal-checkpoint-20261018-0805.json)
contains **20,232 documents** (ECB 11,358; EMA 8,874) and **456,519,243 verified
tokens**, or 45.65% of the target. The ECB audit completed for an older source
revision, while EMA's current audit had counted 8,750 of 8,874 pinned rows. These
are matched cache lower bounds, not a claim of complete current-source coverage.
All limits, private visibility, source pacing and raw-v2 behavior are preserved.

[Six new historical article probes](../research/ecb-article-filename-recovery-20261018.md)
passed both independent identity checks and the production PDF/title validator.
The adapter now recognizes dotted 1999 Monthly Bulletin page-range filenames and
three guarded 2015–2018 Economic Bulletin article styles. Sixty-two remaining
article 404s match the added rules and pass the existing pending/cooldown guard
without failure resets. This is a bounded retry opportunity, not saved yield.
The unmatched focus-style article's exact text is already stored as a box; its
incomplete title match remains rejected rather than counting the same text again.

Validation: **314 tests passed**, plus Ruff lint/format and whitespace checks.
Collector, manual and disabled-schedule recovery apps all deployed successfully.
The running ECB worker can retain its earlier adapter until its next round; no
new-rule ingestion is claimed yet. Neither the one-billion-token objective nor
whole-website exhaustion has been reached.

At 08:10 UTC, [both calls in audit pair 17 were confirmed complete](../research/audit-pair-17-observations-20261018.json).
The [08:08 matching-cache checkpoint](../research/goal-checkpoint-20261018-0808.json)
contains **20,292 documents** and **459,981,239 verified tokens (46.00%)**.
ECB contributed 186,633,399 counted tokens; EMA contributed 273,347,840. Both
summaries complete their pinned audit revisions, while newer collection rows
remain unaudited. The coordinator may retain unresolved bookkeeping until its
next scheduled reconciliation; no extra audit pair was started.

ECB's current campaign remains running. EMA's prior supervisor is now complete
with its full child tree terminated. The existing collection controller can
therefore reconcile and reserve its next bounded campaign; no independent source
worker is introduced. Technical-publication saved rows have reached 58 at the
08:08 checkpoint. All 150 new report-series candidates and all 20 convergence
reports remain completed, with no failures in those five collections.

The [08:13 controller confirmation](../research/continuation-controller-resume-20261018.json)
records **20,332 stored documents** and the same 459,981,239 verified-token lower
bound. The existing controller completed its two-phase terminal reconciliation
using fresh source snapshots, then allocated exactly one EMA successor:
`goal-ema-1792311174-e0a016a7ea`, supervisor `fc-01M5714D7CKZSJ5740VZWY5SWS`.
Its live child tree is running. ECB's existing supervisor is still running.
There are now 26 campaign records, with 6,000 reserved attempts across both sources;
the successor retains three rounds, 1,000 attempts and 3,000 work seconds per round.
No source pacing, schedule, cap or audit reservation was changed.

## Owner request: collect another 10,000 documents

The owner's new request is anchored to a fresh **20,703-document baseline** at
08:38:59 UTC (ECB 11,671; EMA 9,032), making the target **30,703 saved documents**.
[The immutable request](../research/additional-10000-documents-20261018.json)
records both pinned source revisions. It was uploaded to private control path
`milestones/additional-10000-20261018T083859Z.json` and downloaded again to verify
its exact bytes. Only that milestone file was touched; source/control progress,
reservations, schedules, source pacing, caps and raw-v2 storage remain unchanged.

`status.py --document-target <request.json>` now reports additional saved rows,
remaining rows and the actual pinned source revisions. It requires the matching
baseline sources and rejects incomplete, wrong-repository or regressed reads.
A request is reached only at the full additional count. Validation: **323 tests
passed**, with lint/format/whitespace checks passing. This is a read-only local
status change; no collector redeployment or independent writer was needed.

[The first progress check](../research/additional-10000-progress-20261018-0842.json)
contains **20,763 saved documents**: **60 added**, **9,940 still required** for this
request. Exact-token lower bound remains 459,981,239. Both existing collectors
are advancing. ECB's article collection has reached 439 saved rows under the
new adapter, confirming initial ingestion through the expanded article rules.
The latest owner request and command are included at the top of the continuation
instructions so subsequent work retains the same baseline. Neither this
10,000-document milestone nor the overall token goal has yet been reached.


## Owner supersedes the additional-document milestone

The owner now requests collection until documents run out, independently in
Modal. The earlier 10,000-additional-document record is retained as history.
`stop_at_token_target: false` makes one billion verified tokens a progress
milestone rather than a dispatch stop. All existing operating budgets, pacing,
private storage, source reservations and exhaustion evidence requirements remain.
Budget exhaustion still requires review and does not prove website exhaustion.
The change passes 331 tests, including continued dispatch above the token milestone,
strict boolean validation, genuine exhaustion and unchanged document-budget stops.
No notification automation was created.
