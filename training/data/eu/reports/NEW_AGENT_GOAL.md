# New-agent continuation goal


## Latest owner request: continue until documented exhaustion

On 20 October the owner requested queued links/documents and separate parsing.
ECB and EMA now enable `download_queue` in both mappings: persisted inventories
hold links, one paced downloader thread prefetches at most one checkpoint batch,
and the sole HF writer stages original bodies before parsing. Parsed results and
queue removal are atomic; restart and bounded parse retries reuse stored bodies.
These lanes run inside existing bounded Modal workers, not in this chat. Keep
all single-writer, pacing, cooldown, attempt and campaign limits. Raw queue depth
is reported separately from parsed documents and verified tokens.

ECB's previous inventory finished at 19,305 rows. Three more exact official
series (insurance/pension statistics 11, payment statistics 14, SESFOD 454) add
69 distinct candidate URLs with nine successful selected samples. See
`research/ecb-queue-expansion-20261020.{md,json}`. They are candidates until
saved; further coverage discovery remains necessary after this batch. The goal
and manual apps are deployed; current workers may retain the preceding image
until their next round. Verify `document_pipeline` and `document_queue_sha256`
in a fresh run instead of assuming deployment changed an already-running worker.
Live ECB adoption was verified at 07:11 UTC: 20 originals staged before 20 parsed
documents were saved, with zero failures and a matching queue module digest.
See `research/document-queue-live-20261020.json`. The latest catalogue initially
returned a chunk 404; its automatic retry succeeded without inventory injection
or resetting failures. EMA's already-running round retains its prior image;
queued mode is configured for subsequent workers.

The owner superseded the additional-10,000-document request on 18 October 2026:
keep collecting in Modal until documents run out. Neither 30,703 documents nor
one billion verified tokens is a completion condition. Token counts remain an
audited progress metric; `stop_at_token_target: false` disables the token stop.
The earlier immutable milestone remains historical evidence, not an active goal.
See [owner directive](../research/exhaustion-directive-20261018.json).

Preserve every operating budget below. A budget block requires review and is
not evidence that either website is exhausted. Keep existing tracked workers;
collection and parsing run independently of this chat in Modal. No recurring
chat notification has been created (the available automation minimum is hourly,
while the owner had asked for half-hour updates).

The owner then requested immediate EMA continuation and larger batches. EMA
was restarted through the existing controller. Its already running campaign
keeps three rounds; future campaigns use six sequential rounds (up to 6,000
reserved attempts, roughly five work hours). The controller now checks every
minute and performs terminal settlement with two fresh reads in one invocation.
Per-round caps, one writer per source and ten-second EMA pacing remain unchanged.

ECB hit the HF 128-commits/hour repository limit after empty campaigns repeatedly
polled one deferred retry. The controller now backs off empty rounds for at least
ten minutes and honors persisted source cooldowns (`waiting_retry`). The recovery
checks terminal trees and a successful source write before acknowledging repair;
never replay old failed dispatches. Only one configured ECB retry remained then.
Do not confuse configured inventory depletion with whole-website exhaustion;
unconfigured catalog types and unresolved extraction gaps remain.

The subsequent ECB expansion adds four exact FOEDB types: bank-lending surveys
(99), professional-forecaster surveys (158), enterprise-finance surveys (157)
and T2S annual reports (159). The cached snapshot has 198 distinct candidate
URLs, none overlapping prior configured URLs. Eleven of twelve selected samples
extracted; one bank-lending PDF returned 503. Annual accounts are deferred because
historical PDFs are sections of already-collected Annual Reports. Both pipeline
mappings are updated; the goal and manual apps are redeployed. See
`research/ecb-survey-expansion-20261018.json`. Do not count candidates as saved
documents. Preserve discovery of further unconfigured types after this batch.

The owner renewed the request to keep ECB adding new documents on 19 October.
Twelve additional statistical types now provide 2,239 distinct candidate URLs;
all 36 selected full-text samples passed. Both mappings add the series at weight 1
and both Modal apps are redeployed. The existing controller owns the new discovery
campaign; inspect current durable claims before dispatch. See
`research/ecb-statistical-expansion-20261019.md` and its JSON. All 198 preceding
survey/T2S candidates were saved, bringing ECB to 15,382 before this expansion.
Audit scheduling had 42 of 48 pairs reserved at the preceding check; preserve
that budget and distinguish token-accounting limits from collection progress.

The owner subsequently extended the exhausted audit budget on 19 October.
Production now allows **96 audit pairs total**, adding 48 to the original 48.
The grant is recorded in private `tokens/schedule.json` under `budget_extensions`
and in `research/audit-budget-extension-20261019.json`. Existing pair and token
cache history is retained. The audit app is redeployed and its coordinator is
invoked immediately to resume counting. Keep the same serialized audit workers
and per-call shard/runtime limits; collection budgets are unchanged.

The latest owner request is to keep pushing on ECB. A further 22 policy,
infrastructure, survey and digital-euro series now provide 1,687 candidate URLs.
Of 66 selected enabled-series samples, 65 fetched and parsed; one returned 503.
T2S harmonisation and Euro Money Market report types remain deferred because each
had two historical catalog 404s among three selected samples. Do not reset those
failures or guess replacement URLs. Read `research/ecb-policy-project-expansion-20261019.md`
and its JSON. Both mappings/apps are updated; the existing controller owns the
next ECB campaign. Prior survey/statistical expansions are saved; ECB was 17,620
rows before this new work. Preserve the 96-pair audit budget and all collection
limits. Further discovery must continue after this configured backlog drains;
64 configured types in one English metadata snapshot are not website exhaustion.

The owner increased HF private storage after both sources hit an account-level
storage limit. Source writes were verified after the upgrade, terminal failed
campaign metadata was reconciled, and the controller restarted both sources.
New campaigns: ECB `goal-ecb-1792450658-9755390c25` and EMA
`goal-ema-1792450658-4c81e2add9`. Audit measurement resumed at pair 55 of 96.
See `research/hf-storage-recovery-20261019.json`. The pinned pre-restart lower bound
passed one billion at 1,001,931,446 verified tokens; **continue collecting** under
the owner's exhaustion directive. All collection limits and source pacing remain.
Never replay old failed calls or reset completed/failure history after a quota fix.

Paste the following into a new agent/goal:

---

Continue the ECB and EMA document collection in `koogle/lm-exploration` until we have documented genuine exhaustion of both websites. Continue measuring verified OpenAI `o200k_base` tokens as progress. Keep collecting and fixing real failures autonomously; do not stop at a status update or treat a transient service error as website exhaustion.

Start by reading:

- `/workspace/lm-exploration/AGENTS.md`
- `/workspace/lm-exploration/training/data/eu/AGENTS.md`
- `/workspace/lm-exploration/training/data/eu/reports/STATUS_AND_HANDOFF_20261018.md`
- `/workspace/lm-exploration/training/data/eu/README.md`
- `collection_goal.json` and `goal_pipeline_sources.json` in that EU directory.

If this is a new checkout, use repository `https://github.com/koogle/lm-exploration`, branch `feat/eu-collection-followup`, PR #14. Check whether the PR has since merged before choosing the branch. Run EU commands from `training/data/eu`, an isolated `uv` project.

At the handoff snapshot, we had **17,858 stored documents and at least 405,514,647 verified tokens**. Those numbers are historical: load current Hugging Face checkpoints first. ECB and EMA are already collecting on Modal; their schedules continue independently of this chat. Inspect existing reservations and call IDs before launching anything. Never create a duplicate source writer or replay an ambiguous dispatch.

**Keep all datasets private.** The latest owner instruction supersedes earlier public-visibility permission. Durable storage is:

- `jfrick/eu-ecb` and `jfrick/eu-ema`: document Parquet, originals, inventories and progress.
- `jfrick/eu-corpus-control`: collection goal, exact token caches, audit reservations and code backups.
- `jfrick/eu-eurlex` and `jfrick/eu-commission-press`: private historical datasets outside the present ECB/EMA scope.

Use the existing apps: `lm-exploration-eu-100k` for the one-minute collection controller and `lm-exploration-eu-token-audit` for hourly audits. The recovery app's schedule must remain disabled. Source mappings, durable claims and resume logic are already implemented; extend shared tooling rather than inventing separate crawlers. Use subagents for independent source investigations when useful, while preserving source pacing.

Preserve the existing operational envelope: **100,000-document cap, 150 collection campaigns, three rounds per ECB campaign and six per future EMA campaign, at most 1,000 attempts and 3,000 work seconds per round; 96 reserved audit pairs total.** Do not silently raise those caps. Review them before they are exhausted; an exhausted operational budget is not an exhausted website. Preserve the ignored local budget ledger if present. Never overwrite a newer durable ledger with a stale snapshot.

Critical recent fix: ECB reached HF's 10,000-files-per-directory limit. New originals must use **`raw-v2/<hash-prefix>/<hash>.gz`**; preserve existing `raw/<hash>.gz` files and resolve through `raw_blob_metrics` / shared helpers. The live fix passed 252 tests, exceeded 10,000 ECB documents and verified a downloaded original's decompressed SHA-256. Do not flatten new files or move/delete the legacy directory. For a verified repaired source, use explicit recovery acknowledgement only after all its existing campaigns and child workers are confirmed terminal; keep failure history intact.

Continue exact-content deduplication and provenance. Prefer native HTML/XML/TXT, then text-bearing PDFs. Never count catalogs as training text or use LLM rewrites. Respect robots, the five-second ECB and ten-second EMA pacing, and server cooldowns. Retry transient failures at ten-minute intervals when appropriate, but honor longer `Retry-After` values, including HF's observed hourly commit cooldown. Preserve the managed proxy and TLS trust configuration.

Read exact token caches at `tokens/ecb.json` and `tokens/ema.json`, and audit coordination state at `tokens/schedule.json` in the private control repository. Only counts matching current shard hash/path/rows contribute to the token lower bound. Character counts are not tokenizer counts. Audit summaries can be complete for older revisions while collection has advanced; resume audits automatically and verify freshness.

Useful first command: `uv run status.py both --mapping goal_pipeline_sources.json`. Inspect live call trees, investigate actual blockers, and confirm new rows continue to reach HF. The report explains catalog coverage, active-job pointers, new EMA scientific-document partitions, source gaps and safe deployment commands. Neither whole website has been proven exhausted.

Fix and validate issues, commit and push code/research to PR #14 while it is open, or a new PR after it merges. Keep raw data, credentials, Parquet and budget ledgers out of Git. After committing, run `uv run python backup_code.py` and confirm a verified receipt for the committed snapshot in the private control repo. Do not claim a pending backup succeeded. Give concise updates with documents, verified tokens, actual blockers and next actions.

Reaching one billion tokens does not stop collection. Claiming website exhaustion requires documented discovery coverage and resolution of supported inventories and retrieval failures. Report unsupported formats and missing histories as gaps. Do not mark the overall goal complete just because the harness is deployed or a bounded batch finishes.

---
