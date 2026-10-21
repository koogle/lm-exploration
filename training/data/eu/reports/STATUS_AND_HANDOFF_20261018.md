# ECB and EMA corpus: status and agent handoff

Snapshot: **18 October 2026 07:56 Asia/Nicosia** (2026-10-18T04:56:29.578705+00:00 UTC).

The machine-readable companion is [handoff-20261018.json](../research/handoff-20261018.json). Counts were read sequentially from pinned Hugging Face checkpoints. Remote workers continue after this snapshot; refresh before taking action.

## Objective and current result

Continue collecting ECB and EMA documents until the corpus contains **1,000,000,000 verified `o200k_base` tokens**, or both websites are demonstrably exhausted. Neither condition has been reached. Keep all document and control repositories **private**. The original publication ambition remains deferred under the owner’s newer privacy instruction.

Stored corpus: **17,858 documents**. Verified tokenizer coverage: **at least 405,514,647 tokens (40.55% of the target)**. The tokenizer count covers older pinned audit snapshots, so it is a lower bound for the growing corpus.

| Source | Stored documents | Characters | Whitespace words | Verified tokens | Audited documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| ECB | 10,010 | 625,224,571 | 94,528,698 | 172,913,672 | 10,010 |
| EMA | 7,848 | 844,864,079 | 123,781,591 | 232,600,975 | 7,544 |

Additional stored metrics:

- Characters: **1,470,088,650**.
- UTF-8 text: **1,520,764,230 bytes**.
- Zstandard Parquet: **536,809,005 bytes**.
- Compressed original responses: **14,761,487,119 bytes**.

The audit counts each document’s text independently, with no added separators or EOS tokens. It uses `encoding.encode(text, disallowed_special=())`; characters and words are never substituted for tokenizer measurements. Both audit summaries are complete for their respective pinned revisions, but EMA has since added documents. This snapshot is not a new full-corpus integrity audit.

## Repository, PR and durable storage

- GitHub: https://github.com/koogle/lm-exploration.
- Active branch: `feat/eu-collection-followup`.
- Active PR: [#14](https://github.com/koogle/lm-exploration/pull/14), open.
- Original implementation PR #3 was merged into `main`.
- Code revision at this observation: `55c95ee`; this report and handoff are committed afterward on the same PR.
- Work directory: `/workspace/lm-exploration/training/data/eu` (isolated `uv` project).

Private Hugging Face repositories:

| Repository | Role |
| --- | --- |
| `jfrick/eu-ecb` | ECB text Parquet, originals, inventories and source progress |
| `jfrick/eu-ema` | EMA text Parquet, originals, inventories and source progress |
| `jfrick/eu-corpus-control` | Collection goal, audit reservations, token caches and committed-code backups |
| `jfrick/eu-eurlex` | Earlier law collection; outside the current ECB/EMA goal |
| `jfrick/eu-commission-press` | Earlier Commission collection; outside the current goal |

All five repositories were confirmed private during the latest recovery. `pipeline_sources.json` and `goal_pipeline_sources.json` create private repositories. Earlier reports mentioning public visibility describe superseded historical state. Do not republish them.

Code backups live at `code/<git-commit>.tar.gz` plus a checksum manifest in the private control repository. `backup_code.py` archives committed `training/data/eu` files only, uploads them, downloads the pinned archive and verifies its SHA-256. Raw documents, Parquet, credentials, local environments and ignored artifacts are excluded. The local ingestion budget ledger is deliberately ignored; preserve it in this workspace rather than expecting it inside a Git clone or code archive.

## Live execution and budgets

| Modal app | Role |
| --- | --- |
| `lm-exploration-eu-100k` | Fifteen-minute collection controller and bounded source campaigns |
| `lm-exploration-eu-token-audit` | Hourly audit coordinator and serialized token workers |
| `lm-exploration-eu-corpus` | Manual collector entrypoints, also updated with the storage fix |
| `lm-exploration-eu-recovery-20261017` | Recovery controller; schedule disabled, also updated |

The collection goal is active. One organization writer is allowed at a time. A source reservation is persisted before dispatch, call IDs are retained, and ambiguous dispatch gaps are not replayed. Terminal supervisors must have terminated child trees before the controller releases their reservations. A later reconciliation tick reads fresh final source metrics.

The control snapshot has **23 campaign records** and **6,000 reserved document attempts**. Its saved row/token totals can trail the newer source and audit checkpoints; do not treat that lag as lost data.

Bounds retained from the authorized campaign:

- Sources: ECB and EMA only.
- Token target: 1,000,000,000 verified tokens.
- Operational document cap: 100,000; reaching it is not proof of website exhaustion.
- Maximum collection campaigns: 150.
- Each campaign: up to 3 rounds.
- Each round: up to 1,000 attempts, 3,000 work seconds, 3,600-second hard timeout.
- Audit worker: up to 3,000 uncached shards per invocation, 1,500 work seconds, 1,800-second hard timeout.
- Audit coordinator: maximum 48 pairs including the adopted initial pair; 14 pairs reserved at this snapshot. Review before that envelope is exhausted.
- All workers use CPUs.

Active source campaign IDs at observation:

- ECB: `goal-ecb-1792299182-bd9cb0969b`; supervisor `fc-01M56NPBCA8234P7HPEH5YB2C0`; status `running`.

- EMA: `goal-ema-1792292355-491f2ce013`; supervisor `fc-01M56F6187673QTXVSBP9BMK6Q`; status `running`.

These IDs are historical pointers for inspection, not commands to replay. Refresh durable state before any new dispatch.

## Collected source coverage

ECB enumeration uses the website’s versioned FOEDB catalogs. EMA uses official complete JSON catalog snapshots, exact document-type partitions and explicit language URLs. Both preserve source identifiers and fetch provenance. Catalog entries are metadata until full text is successfully retrieved and extracted.

Tables distinguish candidate URLs, saved unique text rows and current failure entries. A completed inventory means the selected catalog subset was enumerated, not that the whole institution was exhausted. Some completed candidates are text duplicates, so candidate completion and unique rows differ. Unsupported-format gaps can keep an inventory’s completeness flag false even after all supported URLs fit within the bound.

### ECB

| Collection | Candidates | Stored unique rows | Failure entries | Inventory complete |
| --- | ---: | ---: | ---: | --- |
| `working_papers` | 3,297 | 3,290 | 7 | True |
| `occasional_papers` | 400 | 400 | 0 | True |
| `economic_bulletin_articles` | 503 | 430 | 73 | True |
| `economic_bulletin_boxes` | 1,829 | 674 | 221 | True |
| `speeches` | 3,077 | 1,872 | 0 | True |
| `press_releases` | 2,713 | 1,862 | 1 | True |
| `interviews` | 617 | 615 | 2 | True |
| `meeting_accounts` | 97 | 97 | 0 | True |
| `decisions` | 248 | 248 | 0 | True |
| `annual_reports` | 36 | 34 | 2 | True |
| `convergence_reports` | 20 | 10 | 10 | True |
| `annual_report_translations` | 253 | 253 | 0 | True |
| `financial_stability_reviews` | 44 | 44 | 0 | True |
| `statistics_papers` | 54 | 54 | 0 | True |
| `research_bulletins` | 127 | 127 | 0 | True |

### EMA

| Collection | Candidates | Stored unique rows | Failure entries | Inventory complete |
| --- | ---: | ---: | ---: | --- |
| `epar_overviews` | 50,656 | 622 | 0 | True |
| `epar_product_information` | 55,216 | 2,472 | 2 | False |
| `epar_assessment_reports` | 1,891 | 1,726 | 3 | True |
| `news` | 3,894 | 598 | 0 | True |
| `scientific_guidelines` | 2,793 | 598 | 1 | True |
| `committee_documents` | 5,177 | 550 | 44 | True |
| `non_epar_documents` | 100,000 | 499 | 2 | False |
| `epar_variation_reports` | 3,404 | 326 | 5 | True |
| `epar_scientific_discussions` | 529 | 276 | 0 | True |
| `epar_scientific_discussion_variations` | 191 | 181 | 0 | True |

ECB paper inventories are nearly fully collected; that is not an institutional exhaustion claim. Recently added FSR, statistics-paper and research-bulletin collections are collected in their current catalog scope. Annual-report translations follow explicit same-report language anchors. German/French annual-report samples contained translated text. FSR language-picker variants are excluded because a German route actually returned English with an unavailable-language notice.

The three new EMA partitions are now producing real documents. Their original offline catalog study identified 4,121 supported PDF candidates with no exact URL overlap against the prior partitions; fresh catalogs can change those counts. Text deduplication has removed substantial repeated scientific-discussion content. Product-information translations remain important for multilingual coverage. Broad `epar_documents` collection is deliberately not enabled alongside the precise partitions.

## Storage, resumption and deduplication

`pipeline.py` discovers or resumes saved inventories, chooses eligible supported candidates with weighted round-robin scheduling, fetches through the shared robots/pacing collector, extracts source text and writes atomic batches through `HFStore`.

`hf_store.py` writes immutable Parquet partitions by collection and format, originals, source progress, indexes, growth metrics and dataset cards in one compare-and-swap commit. It does not overwrite a stale source checkpoint after another writer advances the repository.

Deduplication layers:

1. A durable candidate identity incorporates organization, document identifier, URL, language, source modification value and extraction version. Completed candidates are skipped on resume.
2. SHA-256 of normalized extracted text deduplicates equivalent documents within an organization, including different URLs and format copies.
3. Immutable document/shard identities avoid duplicating committed rows on replay.
4. Raw responses are content-addressed; known hashes are reused.
5. Source-language versions retain their relationships and are intentionally distinct when text differs.

These are exact-content safeguards. Cross-organization deduplication, paragraph overlap detection, semantic/near-deduplication and whole-issue/article containment removal are not implemented. Counts are not a novelty claim or a count of independent facts. The text is extracted directly without an LLM rewrite.

## Overnight ECB failure and confirmed recovery

ECB stopped at 9,997 unique rows because its legacy `raw/` folder reached Hugging Face’s 10,000-files-per-directory limit. The Hub returned HTTP400, and the controller correctly stopped automatic ECB dispatch after repeated failures. EMA continued independently.

The deployed fix writes new originals to `raw-v2/<first-two-hash-characters>/<hash>.gz`. Existing `raw/<hash>.gz` objects remain unchanged. `raw_blob_metrics` is authoritative for locating a hash; `resolve_raw_blob_path()` preserves known legacy locations and `raw_blob_path()` generates new partitioned paths. Metrics migration supports both layouts. Do not move/delete the old raw folder or return to flat new writes.

A new explicit `acknowledge_recovery()` operation records a verified fix on the last failed job only after every source campaign is terminal. It preserves failed job statuses and earlier history, resets the resolved failure streak, and retains the three-new-failures safeguard and all budget caps.

Recovery is proven by a real checkpoint at ECB revision `795ef00f84e11ccf397826989c6683aeaa59a7bb`: 10,010 rows and 13 new partitioned raw objects. A downloaded gzip object was decompressed and its SHA-256 matched the raw hash in its path. The next ECB campaign is active. Evidence: [recovery record](../research/raw-directory-limit-recovery-20261018.json).

## Rate limits, access and outstanding quality gaps

- Hugging Face enforces a 128-commits/hour limit per repository. Frequent token-cache checkpoints previously saturated the control repository. Audits now pace uncached shard requests by two seconds and checkpoint every 50 shards plus their final checkpoint. Respect `Retry-After`, including the observed one-hour cooldown; a ten-minute retry preference does not override a longer server cooldown.
- ECB source interval: five seconds. EMA: ten seconds, retained after earlier rate limiting. Both enforce robots directives and stop a round on HTTP429 with durable cooldown.
- Transient managed-network failures have included HTTP503 and connection failures across unrelated services. Preserve inherited proxy and CA configuration; do not disable TLS or bypass network controls.
- CAS409/412 conflicts protect concurrent writes. Audit cache retries may rebase unrelated control updates only when that audit path remains unchanged. Never overwrite another writer’s source state.
- Some legacy ECB links are genuinely broken or unsupported. Conservative monthly-article and box recovery requires narrowly recognized filenames and PDF opening-title validation. Box candidates are limited to 2014–2018 and at most 20 pages. Exact convergence replacements are hash-pinned.
- Image-only scans, OCR, ASR and audiovisual collection are not implemented. Neither DOC/DOCX nor spreadsheet/container extraction is a supported corpus path.
- The current 100k non-EPAR discovery bound omits some older supported URLs; known unsupported formats remain explicit gaps. Do not label them exhausted.
- Source language labels often come from publisher metadata/URLs; UI language alone is not evidence of body language.
- Comprehensive boilerplate stripping, PII review, licensing/redistribution review and publication-quality near-deduplication remain future work. The current output remains private.
- A paid ChatGPT/Codex usage banner is separate from Modal/HF service limits. Existing remote schedules have continued when the assistant’s UI was paused.

## Validation and evidence

The latest code change passed **252 tests**, Ruff lint, formatting and whitespace checks. New storage tests cover mixed legacy/partitioned data, prefix validation, known-hash reuse and pinned metric migration. Recovery tests preserve failure history, reject unresolved jobs and restore the subsequent failure safeguard. Older full-shard audits and newer exact token audits are recorded separately; do not claim a new full corpus audit just because these tests pass.

Useful research files:

- `research/raw-directory-limit-recovery-20261018.json` — actual storage failure and live recovery.
- `research/automatic-audit-schedule-20261017.json` — durable audit adoption and deployment.
- `research/token-goal-deployment-20261017.json` — token-stop guard.
- `research/ecb-next-expansion-20261017.{md,json}` — added ECB series and FSR language negative result.
- `research/ecb-box-recovery-20261017.{md,json}` — eight title-verified historical boxes.
- `research/ema-targeted-expansion-20261017.{md,json}` — additional exact EPAR types.
- `research/ema-yield-20261017.md` — why candidate counts and token yield differ.
- `research/handoff-20261018.json` — this snapshot’s pinned source revisions, metrics and job pointers.

## Runbook for the next agent

Read root and EU `AGENTS.md`, this report, `README.md`, `collection_goal.json` and the two source mappings first. Preserve ignored budget ledgers and existing work. Use current HF state as truth, not old chat counts. All commands below run from `training/data/eu`.

```sh
uv sync --locked
uv run status.py both --mapping goal_pipeline_sources.json
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
```

For exact audit state, read private control paths `tokens/ecb.json`, `tokens/ema.json` and `tokens/schedule.json` at one pinned control revision. Count only cache entries whose source shard hash, path and rows match current corpus state; `goal_controller.cached_token_lower_bound()` implements that check.

Inspect supervisor and child trees using `modal_goal.inspect_call()` and inspect audits using `modal_token_audit.inspect_audit()`. A timeout or unknown dispatch result is not permission to spawn another writer. The scheduled apps already own continuation. If a tick needs a manual kick, use the existing `lm-exploration-eu-100k.tick` or audit `scheduled_tick`, retain its call ID before waiting, and inspect durable results.

Deploy only when code changes warrant it:

```sh
uv run python modal_cli.py deploy modal_goal.py
uv run python modal_cli.py deploy modal_token_audit.py
# Manual collector entrypoints, when shared writer code changes:
uv run python modal_cli.py deploy modal_pipeline.py
```

The CLI wrapper preserves the managed CA for Modal. Do not bypass the wrapper to work around TLS. The recovery app accepts `EU_MODAL_APP_NAME` and `EU_GOAL_SCHEDULE=0`; keep its schedule disabled. Do not create a second independent scheduled collector for the same source.

Commit/push changes through PR #14 while it remains open; use a new branch/PR for subsequent changes if it has been merged. Back up committed EU code with `uv run python backup_code.py`. Verify its receipt rather than claiming a queued backup succeeded.

## Priority work and completion criteria

1. Confirm both sources remain active, the raw-v2 fix reaches subsequent EMA workers, and current token coverage grows.
2. Let the durable queues run; investigate actual failures and unsupported inventories without resetting successful candidates.
3. Ensure audit coverage stays current and review the remaining audit/campaign envelopes before they are exhausted. Operational limits are not website exhaustion.
4. Expand distinct verified ECB/EMA collections when current coverage warrants it; avoid duplicate umbrellas and misleading language variants. Validate retrieval and extraction before claiming token yield.
5. When the matched-shard exact lower bound reaches one billion, stop new collection campaigns, let tracked active work settle, refresh the pinned count and verify the endpoint. Do not mark success based on characters.
6. If token target remains unmet, website exhaustion requires documented discovery coverage and resolved supported inventories, not merely no remaining candidates in the present mapping. The controller intentionally does not infer institution-wide exhaustion from catalog completeness.
7. Produce a final private dataset inventory and evidence-based token report. Keep all data private unless the owner explicitly changes that instruction.

A ready-to-paste continuation request is in [NEW_AGENT_GOAL.md](NEW_AGENT_GOAL.md).
