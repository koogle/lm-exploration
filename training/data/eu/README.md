# EU institutional corpus exploration

Build a continuously maintained, multilingual collection of EU-produced content:
laws, press releases, reports, committee papers, proceedings, meeting notes,
speeches and eventually press-conference/hearing transcripts. Publish curated,
versioned Parquet releases under the owner's Hugging Face handle after the
discovery and extraction approach is measured.

The hypothesis is that agents can find useful content behind fragmented department,
subgroup and meeting pages, discover better bulk interfaces, and maintain collection
adapters. Existing EUR-Lex/OPUS corpora are overlap baselines, not evidence that the
institutional long tail has been collected. No claim of novelty or overall token
volume is established by this pilot.

Findings, measured totals, coverage tables and negative results are in
[RESEARCH.md](RESEARCH.md).

## Implemented now

- Ten collection candidates in [sources.json](sources.json), ordered by priority.
- A bounded collector for HTML/XHTML, selected full-document XML, TXT and
  text-bearing PDF, plus HTML/RSS/Atom/sitemap discovery (`pilot.py`, `eu_corpus.py`).
- ECB, EMA, Eur-Lex and Commission press adapters, durable Hugging Face
  checkpoints and bounded CPU workers on Modal (`pipeline.py`, `modal_pipeline.py`).
- A durable download queue (`document_queue.py`) separating paced downloads from
  parsing inside each worker.
- A scheduled goal controller (`modal_goal.py`) and token audits (`modal_token_audit.py`).

No harness invokes an LLM or browser autonomously. Autonomous adapter repair, ASR,
OCR and near-deduplication remain future stages.

## Ten starting collections

| Priority | ID | Content | Initial scope |
| --- | --- | --- | --- |
| 1 | `eurlex` | Published laws, regulations, directives, decisions | CELLAR native XHTML sample; bulk inventory still needed |
| 2 | `commission-press` | Commission press releases, speeches, statements, Q&A | RSS works; detail pages need an adapter |
| 3 | `ecb` | Central bank releases, speeches, decisions, meeting accounts | RSS and native HTML |
| 4 | `esma` | Securities/markets news, reports, consultations | Native HTML and linked documents |
| 5 | `ema` | Medicines news, guidance, assessments, committee material | Native HTML plus JSON inventory discovery |
| 6 | `eib` | Investment Bank releases and research | Candidate, not yet collector-validated |
| 7 | `parliament` | Committee papers, proceedings, questions, research | API and meeting-page exploration |
| 8 | `council` | Public papers, working-party material, conclusions, press | Public register and press exploration |
| 9 | `expert-groups` | Expert subgroup and comitology meeting material | Group → subgroup → meeting → attachment |
| 10 | `audiovisual` | Press conferences, briefings, hearings, proceedings | Media/transcript inventory; no ASR yet |

ESMA is our working interpretation of “European market agency”; both ECB and EIB
are included for the bank category. EFSA, EEA, ECDC, EBA, EIOPA, Eurofound, Eurostat,
JRC, CORDIS and further agencies are expansion candidates, not silently counted
as covered by this list.

## Run from this folder

This is an isolated uv project; it does not install PyTorch or alter the training
UI registry. Python 3.12 and the checked-in lockfile are used.

```sh
cd training/data/eu
uv sync --locked

# Produce five JSONL task briefs for parallel research agents; no network access.
uv run pilot.py --limit 5 --plan-only

# Collect a bounded slice and export documents.parquet in an immutable run directory.
uv run pilot.py --limit 5 --max-pages 6 --max-depth 1

# Investigate the second half with the same task contract.
uv run pilot.py --offset 5 --limit 5 --plan-only

# Run one source with explicit bounds and destination.
uv run eu_corpus.py crawl --source ecb --max-pages 6 --max-depth 1 \
  --output artifacts/ecb-example
uv run eu_corpus.py export --input artifacts/ecb-example \
  --output artifacts/ecb-example/documents.parquet

uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
```

`--max-pages` bounds attempted content URLs **per source**, including failures;
robots fetches and redirect hops are additional. HTTP bodies have a 20 MB default
cap and requests have a 30-second timeout. The collector stays within each
source's exact host allowlist, checks robots before content requests, spaces
requests by at least one second and honors supported robots crawl delays. No
authentication or browser challenge bypass is implemented. Failed robots access
is recorded and stops that origin; 404/410 robots responses allow normal crawling.

The original `pilot.py` and `eu_corpus.py` commands require a new output directory
and fetch again on repeated runs. Their generic crawler does not traverse arbitrary
JSON inventories. ETags and Last-Modified values are recorded but conditional HTTP
requests are not implemented. The ECB/EMA pipeline below adds explicit inventory
adapters and durable resume; its behavior differs from these original commands.

## Durable collection pipeline

[pipeline_sources.json](pipeline_sources.json) is the single organization → HF
repository → adapter mapping used by manual runs, the goal controller and token
audits. A source's collections default to its adapter's `COLLECTIONS` registry
minus `exclude_collections`; list `collections` explicitly only to select a subset
(Eur-Lex years). `collection_weights` lists only weights other than 1.
`pipeline.load_mapping` resolves the defaults. Repositories are created private.

`pipeline.py` enumerates and saves each collection, then interleaves candidates
round-robin by weight so one large collection does not consume the budget. The
limit is **1,000 attempted documents per organization per round**, including
failures; inventory requests are additional. A worker gets 3,000 work seconds, a
3,600-second Modal timeout, one CPU and 2 GiB memory.

### Download queue

Every source runs documents through `document_queue.py`:

1. One paced downloader thread fetches the next batch (20 items) into a local
   spool. It never writes HF and stops at the first HTTP 429.
2. The sole writer commits the gzip originals plus `download_queue` entries, then
   starts the next download and parses the staged batch.
3. Parsed Parquet, the dedup index and queue removal commit atomically.
4. On restart, staged bodies are parsed before any new download (no refetch).

Parse failures keep the staged body and retry after 300 s × attempt, up to three
attempts; exhausted items leave the queue and keep `raw_path` in their failure
record. An unreadable staged body is recorded as a failure and dropped from the
queue so a later worker downloads it again. At most one batch is downloaded
speculatively; an interrupted run reserves one batch against its budget.
`status.py` reports queue depth separately from parsed documents.

### Run, resume, and refresh

Authenticate Hugging Face locally with a token authorized to create and write
these dataset repositories. Configure a Modal secret named
`eu-corpus-huggingface` containing `HF_TOKEN`; never commit credentials. Modal's
own account authentication must also be configured.

```sh
# Local worker: writes to the mapped HF repository, not just local artifacts.
uv run pipeline.py ecb --run-id ecb-pilot-001 --max-documents 1000 --work-seconds 3000

# Remote credential and source-access checks, without starting collection.
uv run python modal_cli.py run modal_pipeline.py --check-only

# Start both CPU workers; keep the coordinator alive until both finish.
uv run python modal_cli.py run --detach modal_pipeline.py --run-id eu-pilot-001 \
  --source both --max-documents 1000 --work-seconds 3000

# Resume one organization using its saved inventories and completed-item state.
# Use a new run ID: the local reservation ledger rejects duplicate run IDs.
uv run python modal_cli.py run --detach modal_pipeline.py --run-id eu-pilot-002 --source ema

# Re-enumerate current inventories, retaining progress for known document revisions.
uv run python modal_cli.py run --detach modal_pipeline.py --run-id eu-refresh-001 --source both --refresh
```

No recurring schedule is installed by these commands. Single-round commands run
one bounded batch; `--rounds 3` launches a finite continuation campaign. The
1,000-document limit is not a lifetime repository cap.
The Modal entrypoint checks the HF identity against the configured owner and tests
both institutions' robots endpoints before collection. It reserves compute in
`artifacts/modal-ingestion-budget.json`, independently of training experiments,
and records call IDs and returned summaries under `artifacts/`. Reservations are
conservative limits, not measured bills. Preserve this local ledger between sessions.
Only one writer per organization should run at a time; remote functions are limited
to one container each, and HF compare-and-swap protects against overlapping writers.

### Durable layout and crash recovery

| HF path | Contents |
| --- | --- |
| `data/<collection>/<format>/part-<hash>.parquet` | Extracted text, stable schema, document/revision IDs and provenance |
| `state/progress.json` | Completed items, failures, retry times, dedup index, download queue, inventory paths, source mapping and run summaries |
| `inventories/<collection>/<hash>.json` | Enumerated document-language candidates and coverage evidence |
| `raw-v2/<prefix>/<sha256>.gz` | Compressed original bodies, staged before parsing; legacy `raw/<sha256>.gz` files are retained |
| `README.md` | Collection configs, stored row counts and unreviewed/mixed source-term status |

Collection configs use `data/<collection>/**/*.parquet` as their `train` split;
formats are directory partitions within each collection. Each stored row includes
source URL, document ID, language, dates, extracted text, character/word counts,
raw/text hashes, extractor version and JSON provenance. Raw bodies for rejected
or exact-duplicate documents and complete HTTP logs are not all persisted to HF;
local worker artifacts are ephemeral on Modal.

A checkpoint commits new Parquet shards, their raw bodies, progress and the dataset
card together. Enumeration checkpoints atomically save an inventory with its state
reference. Document checkpoints occur every 20 attempts and at normal completion.
Commits require the previously loaded HF revision; a conflict aborts the worker
rather than overwriting another writer's progress. An interrupted worker resumes
from the last committed checkpoint. Uncommitted downloads may repeat; committed
items are skipped. A stale `running` entry is evidence of interruption, not a claim
that the worker is still alive. Application-level Modal retries are disabled. Platform interruptions can still
restart a worker; durable progress determines which documents remain.

Completed-item keys include source, document ID, URL, language, known modification
date and extractor version. Exact normalized-text duplicates point to the existing
row and retain their source mapping in progress. The dedup index spans collections
within an organization, so a duplicate may reference a row in another collection.
Unchanged text after an extractor upgrade can likewise retain the original row's
extraction-version metadata. Deduplication does not span organizations.

Download/extraction failures are saved with up to three attempts per item and a
retry delay of 300 seconds times the attempt count. The worker does not sleep until
a retry becomes eligible; a later invocation retries it. Items at the attempt cap
require deliberate state repair or a new item revision before retrying. Do not
modify progress independently of the data it references.

`--refresh` reloads inventories; it does **not** unconditionally redownload completed
URLs. In particular, ECB inventory records currently lack a modification timestamp,
so content changes at an existing URL can go undetected. Periodic forced revalidation,
conditional requests, deletions, cross-document version relationships, DOC/DOCX
extraction, OCR, automatic repair and scheduled collection are not implemented.
Inventories are bounded to 100,000 candidates per collection; read their recorded
coverage limitations before calling a backfill complete. Counts are not tokenizer
measurements or evidence of dataset novelty.

## Research-agent workflow

1. Parent selects a registry slice and assigns one collection per task. Read
   RESEARCH.md first; keep agents on separate report files.
2. Agent inspects the institution and its linked collections, feeds, public APIs,
   pagination, language alternatives and attachments. Use bounded live probes.
3. Agent writes `artifacts/research/<id>.md` and `.json` (ignored): tested URLs,
   UTC timestamps, HTTP statuses, full-text evidence, recommended configuration
   and unresolved gaps. Raw material stays in `artifacts/source-probes/<id>/`.
   The parent folds concise findings into RESEARCH.md.
4. Parent reviews and integrates source rules or a collection adapter, runs the
   bounded pilot, and examines extracted bodies rather than trusting HTTP 200.
5. Use failures and uncovered collections to assign the next investigation.
   Parent commits and pushes code and findings; agents do not race Git writes.

Agent-generated text is for research notes, not corpus contents. Extract the
original document; never paraphrase it into a supposedly native training example.

## Outputs and provenance

The original local discovery/pilot commands write the following artifacts; the
HF-backed pipeline uses the durable layout above:

| File | Meaning |
| --- | --- |
| `run.json` | Source configurations, bounds, timestamp, collector hash/version |
| `raw/<sha256>` | Exact downloaded response bodies, including inventories |
| `attempts.jsonl` | Fetch outcomes, errors, HTTP status and response metadata |
| `provenance.jsonl` | Successful retrieval records and raw hashes |
| `discoveries.jsonl` | Parent-child links, classification, skips and failures |
| `documents.jsonl` | Extracted text and record-level provenance |
| `summary.json` | Observed counts before cross-document deduplication |
| `documents.parquet` | Exact normalized-text-deduplicated records |
| `export-summary.json` | Post-deduplication counts from `pilot.py` |
| `agent-tasks.jsonl` | Source investigation briefs pointing back to the run |

Parquet retains text, title, language, character/whitespace-word counts, URL aliases,
source IDs and JSON provenance. A content hash identifies exact normalized text;
this is not a substitute for a future work/language/version identifier. Language
currently comes from document markup or explicit source fallback, not language
identification; untranslated templates can mislabel multilingual text.

Raw files and corpus outputs are ignored by Git. Commit source definitions,
investigation evidence, tests, code and compact measured summaries. Do not add this
candidate to `../datasets.json` as an eligible training corpus. Source notices are
retained as metadata during exploration; publication review remains separate.

## Quality and sizing

Native structured full text is preferred over PDFs, not specifically the `.txt`
extension. CELLAR supports XHTML and Formex XML. Metadata XML, RSS summaries and
API wrappers are inventories, not training documents. Generic page extraction is
an initial approximation: inspect article boundaries, navigation, repeated blocks,
tables and footnotes before claiming training readiness. Near-duplicates,
revised laws, translated versions and mirrored publications still need explicit
grouping. Preserve paragraphs in stored text; normalize whitespace only for hashing.

Counts are **whitespace words and characters, not model tokens**. A source-level
token forecast needs a named tokenizer, a representative language/document sample,
an inventory count and measured extraction/deduplication yields. Our previous
20–50B-token discussion was a provisional broad planning hypothesis, not a measured
result or a cap. Do not extrapolate from six recent pages or the long GDPR sample.

## Continuous collection and media: next stages

- Discover with agents; run stable adapters with cheap workers; return failures
  and new subcollections to the discovery queue.
- Extend the implemented HF progress registry and raw snapshots with change
  feeds, conditional HTTP and inventory diffs; reconcile periodically for missed,
  backdated, changed or removed records. CELLAR publishes change notifications.
- On Modal, separate browser discovery, CPU downloads/extraction and GPU ASR/OCR.
  Persist work state outside ephemeral queues. Measure requests, bytes and compute
  per source before choosing budgets and schedules; do not reuse research ledgers
  or resources from the latent-workspace project.
- Link agenda, prepared statement, minutes, recording, captions and transcript
  through an event ID. First retrieve existing text/captions; transcribe missing
  speech, especially Q&A. Preserve timestamps, speaker evidence, language, original
  versus interpreted tracks, ASR model/version and transcript origin. An official
  caption may be automatic; a prepared speech is not a verbatim transcript.
- Publish immutable HF snapshots plus documented additions/corrections. Split
  related versions/translations by work/event to avoid evaluation leakage.
- Expand country by country into parliaments, ministries, agencies and regional
  government. Germany's Bundestag, France's Assemblée nationale/DILA and Spain's
  BOE provide structured entry points, not complete national coverage.

## Primary references

- [CELLAR publication retrieval](https://op.europa.eu/en/web/cellar/cellar-data/publications)
  and [change feeds](https://op.europa.eu/en/web/cellar/cellar-data/rss-and-atom-feeds).
- [EMA JSON inventories](https://www.ema.europa.eu/en/scientific-guidelines/download-website-data-json-data-format).
- [Parliament committee meeting documents](https://www.europarl.europa.eu/committees/en/meetings/meeting-documents).
- [Comitology dataset exports](https://ec.europa.eu/transparency/comitology-register/screen/datasets).
- [Commission audiovisual FAQ](https://audiovisual.ec.europa.eu/en/faq)
  and [Parliament multimedia services](https://www.europarl.europa.eu/news/en/media-services/multimedia-centre).
- [Eurlex Resources overlap baseline](https://huggingface.co/datasets/joelniklaus/eurlex_resources).
- [Bundestag open data](https://www.bundestag.de/services/opendata),
  [French parliamentary debates](https://data.assemblee-nationale.fr/travaux-parlementaires/debats),
  [BOE API](https://www.boe.es/datosabiertos/api/api.php).

### Inspect durable progress and count tokens

```sh
uv run status.py both
uv run status.py both --audit --tokens > artifacts/hf-audit.json
```

The audit pins each repository revision, downloads its Parquet shards, validates
IDs and text hashes, and counts actual text tokens with OpenAI `o200k_base`. It
does not estimate the uncrawled corpus. Status and audit commands do not write to HF.
Modal detached local entrypoints only guarantee survival of their last triggered
function after disconnection; keep the two-source coordinator alive. If either
worker stops, rerun with a new run ID to resume from its HF checkpoint.

## Campaigns and the goal controller

Launch each source separately with a unique run ID; never start a second writer
for an active source. A campaign runs at most three rounds (EMA six), respects
the saved source cooldown, and persists each child `call_id` before the worker
starts so a restarted supervisor reattaches instead of dispatching again.

```sh
uv run python modal_cli.py run --detach modal_pipeline.py --source ecb \
  --run-id ecb-expansion-001 --rounds 3
# Wait for a predecessor campaign (default 30 min, at most 3 h).
uv run python modal_cli.py run --detach modal_pipeline.py --source ema \
  --run-id ema-next-001 --rounds 3 --after-campaign PREVIOUS_CAMPAIGN_ID
uv run status.py all
```

`modal_goal.py` deploys `lm-exploration-eu-100k`, a one-minute controller driven by
[collection_goal.json](collection_goal.json). It launches bounded ECB/EMA
campaigns, one active campaign per source, with CAS claims, reservations and call
IDs in the private `jfrick/eu-corpus-control` repository. An uncertain dispatch
blocks its source until reconciled; three consecutive failures stop dispatch
until an explicit `acknowledge_recovery`. A zero-attempt round with remaining
items imposes a ten-minute backoff. The 150-campaign ceiling is a blocked budget
state, not success.

```sh
uv run python modal_cli.py deploy modal_goal.py
# Recovery app with the schedule disabled; it shares control state and claims.
EU_MODAL_APP_NAME=lm-exploration-eu-recovery-20261017 EU_GOAL_SCHEDULE=0 \
  uv run python modal_cli.py run --detach modal_goal.py
```

To halt dispatch, stop the deployed app; stop campaigns in other apps separately
and reconcile their terminal call trees before resuming. Never clear unresolved
claims to force a launch.

## Token audits

`token_audit.py` counts `o200k_base` tokens per stored text (no separators or EOS)
at a pinned revision, verifying each shard's row count and record digest, and
caches per-shard results in `tokens/<source>.json` in the control repo.
`modal_token_audit.scheduled_tick` runs hourly within the 96-pair budget, with
a 1,500-second work budget, 2-second pacing per uncached shard and HTTP 429
cooldowns.

```sh
uv run python modal_cli.py deploy modal_token_audit.py
uv run python modal_cli.py run --detach modal_token_audit.py --source ecb --max-shards 3000
uv run status.py both --cached-tokens
```

`--cached-tokens` matches cached counts to current shards by hash, path and row
count without downloading Parquet; missing caches are reported, not zeroed.

## HF storage behaviour and code backups

Transient HF reads (network errors, 502/503/504) get at most three attempts. A
transient commit error counts as success only if a fresh pinned head contains
exactly the intended progress bytes; HF's "another commit in progress" 409 is
retried up to three times on the unchanged parent. Stale-parent conflicts
(409/412) abort rather than overwrite another writer.

`uv run backup_code.py` uploads the committed `training/data/eu` subtree to the
control repo as `code/<git-commit>.tar.gz` with a checksummed manifest and
verifies the upload. `uv run retry_backup.py --interval 600` retries until a
verified upload succeeds.
