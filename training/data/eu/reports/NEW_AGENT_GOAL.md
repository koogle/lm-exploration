# New-agent continuation goal


## Latest owner request: continue until documented exhaustion

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

Use the existing apps: `lm-exploration-eu-100k` for the fifteen-minute collection controller and `lm-exploration-eu-token-audit` for hourly audits. The recovery app's schedule must remain disabled. Source mappings, durable claims and resume logic are already implemented; extend shared tooling rather than inventing separate crawlers. Use subagents for independent source investigations when useful, while preserving source pacing.

Preserve the existing operational envelope: **100,000-document cap, 150 collection campaigns, three rounds per campaign, at most 1,000 attempts and 3,000 work seconds per round; 48 reserved audit pairs total.** Do not silently raise those caps. Review them before they are exhausted; an exhausted operational budget is not an exhausted website. Preserve the ignored local budget ledger if present. Never overwrite a newer durable ledger with a stale snapshot.

Critical recent fix: ECB reached HF's 10,000-files-per-directory limit. New originals must use **`raw-v2/<hash-prefix>/<hash>.gz`**; preserve existing `raw/<hash>.gz` files and resolve through `raw_blob_metrics` / shared helpers. The live fix passed 252 tests, exceeded 10,000 ECB documents and verified a downloaded original's decompressed SHA-256. Do not flatten new files or move/delete the legacy directory. For a verified repaired source, use explicit recovery acknowledgement only after all its existing campaigns and child workers are confirmed terminal; keep failure history intact.

Continue exact-content deduplication and provenance. Prefer native HTML/XML/TXT, then text-bearing PDFs. Never count catalogs as training text or use LLM rewrites. Respect robots, the five-second ECB and ten-second EMA pacing, and server cooldowns. Retry transient failures at ten-minute intervals when appropriate, but honor longer `Retry-After` values, including HF's observed hourly commit cooldown. Preserve the managed proxy and TLS trust configuration.

Read exact token caches at `tokens/ecb.json` and `tokens/ema.json`, and audit coordination state at `tokens/schedule.json` in the private control repository. Only counts matching current shard hash/path/rows contribute to the token lower bound. Character counts are not tokenizer counts. Audit summaries can be complete for older revisions while collection has advanced; resume audits automatically and verify freshness.

Useful first command: `uv run status.py both --mapping goal_pipeline_sources.json`. Inspect live call trees, investigate actual blockers, and confirm new rows continue to reach HF. The report explains catalog coverage, active-job pointers, new EMA scientific-document partitions, source gaps and safe deployment commands. Neither whole website has been proven exhausted.

Fix and validate issues, commit and push code/research to PR #14 while it is open, or a new PR after it merges. Keep raw data, credentials, Parquet and budget ledgers out of Git. After committing, run `uv run python backup_code.py` and confirm a verified receipt for the committed snapshot in the private control repo. Do not claim a pending backup succeeded. Give concise updates with documents, verified tokens, actual blockers and next actions.

Reaching one billion tokens does not stop collection. Claiming website exhaustion requires documented discovery coverage and resolution of supported inventories and retrieval failures. Report unsupported formats and missing histories as gaps. Do not mark the overall goal complete just because the harness is deployed or a bounded batch finishes.

---
