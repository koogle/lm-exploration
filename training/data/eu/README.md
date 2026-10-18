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

## Current status and agent handoff

See the [detailed status report](reports/STATUS_AND_HANDOFF_20261018.md) and
[ready-to-paste continuation goal](reports/NEW_AGENT_GOAL.md). Those are pinned
handoff snapshots; refresh private HF state for live totals. See also the
[18 October continuation](reports/CONTINUATION_20261018.md) for subsequent repairs
and a newer pinned checkpoint.

## Implemented now

- Ten collection candidates in [sources.json](sources.json), ordered by priority.
- Five source investigations with live HTTP evidence in [research/](research/).
- A bounded collector for HTML/XHTML, selected full-document XML, TXT and
  text-bearing PDF, plus HTML/RSS/Atom/sitemap discovery.
- A repeatable loop over a slice of the registry, raw-response provenance, explicit
  failures, exact-text deduplication and Zstandard-compressed Parquet export.
- Portable per-source agent task briefs, with separate report paths so research
  agents can work in parallel and the parent can integrate their findings.
- ECB and EMA collection adapters, durable Hugging Face checkpoints, and bounded
  CPU workers on Modal. Source-to-repository mappings live in code.

The [first measured pilot](research/pilot-2026-10-16.md) produced **14 unique
documents, 491,330 characters and 76,499 whitespace words** from four of the five
investigated sources. Commission Press Corner needs a full-text adapter. These
are small-sample observations, not a token forecast.

The research agents in the initial implementation were delegated by the coding
assistant. `pilot.py` itself is a deterministic collection loop: it does not invoke
an LLM or a browser autonomously. Task briefs are inputs to an agent host such as
Codex, not an invented agent API. `pipeline.py` runs the integrated ECB/EMA adapters
and uploads atomic checkpoints to Hugging Face; `modal_pipeline.py` runs those
workers remotely. Neither harness autonomously invokes an LLM. Autonomous adapter
repair, scheduled refresh, ASR, OCR and near-deduplication remain future stages.
The implementation below is not itself evidence of a successful cloud run; record
measured run outcomes separately.

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

## ECB/EMA durable collection pipeline

[pipeline_sources.json](pipeline_sources.json) is the authoritative organization →
HF repository → adapter → collection mapping. Document repositories are created private following the owner's current instruction;
existing repository visibility is preserved. Initial destinations and configs:

| Source | Hugging Face repository | Collection configs |
| --- | --- | --- |
| ECB | `jfrick/eu-ecb` | `speeches`, `press_releases`, `interviews`, `meeting_accounts`, `decisions` |
| EMA | `jfrick/eu-ema` | `news`, `scientific_guidelines`, `committee_documents` |

The ECB adapter resolves the versioned FOEDB inventory used by its publications
website, enumerates type indexes, and prefers HTML over PDF for each explicitly
listed language. It does not enumerate all other language databases or linked
attachments. The EMA adapter reads official JSON inventories, partitions records
by document category, and expands listed translations. Inventory completeness is
reported relative to these selected interfaces, not the whole institution.

`pipeline.py` enumerates and saves each collection, then interleaves candidate
lists round-robin so one large collection does not consume the entire budget.
The limit is **1,000 attempted documents per organization per round**,
including failed downloads/extractions. Inventory requests are additional. A source
worker gets 3,000 seconds of work time and a 3,600-second Modal timeout; a request
already in progress can run beyond the work deadline. Each worker uses one CPU and
2 GiB memory, with no GPU. These are upper bounds, not promised yields.

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
| `state/progress.json` | Completed items, failures, retry times, dedup index, collection inventory paths, source mapping and run summaries |
| `inventories/<collection>/<hash>.json` | Enumerated document-language candidates and coverage evidence |
| `raw-v2/<prefix>/<sha256>.gz` | New compressed original bodies; existing `raw/<sha256>.gz` files are retained |
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

1. Parent selects a registry slice and assigns one collection per task. Read the
   existing source report first; keep agents on separate report files.
2. Agent inspects the institution and its linked collections, feeds, public APIs,
   pagination, language alternatives and attachments. Use bounded live probes.
3. Agent writes `research/<id>.md` and `.json`: tested URLs, UTC timestamps, HTTP
   statuses/content types, full-text evidence, recommended source configuration,
   and unresolved gaps. Raw material stays in `artifacts/source-probes/<id>/`.
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

### First Modal pilot: 16 October 2026

The [checkpoint report](research/modal-pilot-2026-10-16.json) records actual HF
readback: 20 ECB rows and 139 EMA rows passed shard, text-hash, unique-ID and
progress-reference checks. ECB was still running at that snapshot. EMA was
stopped after persistent HTTP 429 responses; 500 attempts (139 successes,
361 failures) were durably recorded, with possible additional uncommitted work.
The initial implementation continued after 429s; that behavior is now fixed.
Future runs stop on the first 429, atomically save pending work, honor Retry-After
(or a 15-minute fallback), and refuse restarts during the durable source cooldown.
EMA now has a five-second minimum request interval. The fix passed offline tests;
a corrected live EMA run has not yet been attempted.

The selected inventories contain 6,752 ECB and 11,864 EMA document-language URLs.
Those are candidate counts, not extracted-document or token totals. Some document
footers and related links remain in extracted text. Token counting requires the
OpenAI vocabulary file; its download was denied by this environment proxy (403).

## Running totals and expanded collection campaigns

Every HF commit now records exact `metrics` and `collection_metrics`: stored rows,
Unicode characters, whitespace words, UTF-8 text bytes, Parquet file bytes, and
compressed original-file bytes. `metrics_history` retains timestamped growth
checkpoints. HF dataset cards show current totals and a per-collection table;
`uv run status.py both` reports both organizations and combined totals without
redownloading corpus files. Historical rows are backfilled from pinned Parquet
and HF object-size metadata before counters are enabled. These are stored-data
counts, not token counts, network traffic, or total HF repository billing size.
Shared raw blobs count once; inventories, progress and Git history are excluded.

Added collections:

| Organization | Collection IDs | Observed inventory |
| --- | --- | --- |
| ECB | `working_papers`, `occasional_papers`, `economic_bulletin_articles`, `economic_bulletin_boxes` | 6,032 publication records in the observed snapshot |
| EMA | `epar_overviews`, `epar_product_information`, `epar_assessment_reports` | 50,656 / 55,214 / 1,891 accepted PDF language URLs |

EMA EPAR collections include 26 languages. They are scheduled instead of the
umbrella EPAR collection to avoid overlapping collection lists. Historic ECB PDFs
without a language suffix are retained as `und`, not assumed English. See
[ECB expansion](research/ecb-expansion.md) and [EMA expansion](research/ema-expansion.md).

Launch each source separately so a detached local entrypoint has one durable
supervisor call. Each campaign allows at most three rounds of 1,000 attempted
documents; each collection worker has a 3,600-second hard timeout and 3,000-second
work budget. Supervisors have a four-hour deadline and low CPU allocation. They
respect the saved source cooldown before another round, and stop if a crashed
supervisor cannot determine whether a claimed worker already ran. Plans, claims,
round results and metrics live in HF `state/progress.json` under `campaigns`.

```sh
uv run python modal_cli.py run --detach modal_pipeline.py --source ecb \
  --run-id ecb-expansion-001 --rounds 3
uv run python modal_cli.py run --detach modal_pipeline.py --source ema \
  --run-id ema-expansion-001 --rounds 3
uv run status.py both
```

EMA requests now have a ten-second minimum interval; ECB uses five seconds.
A 429 ends the current round immediately. There is no indefinite or recurring schedule: campaigns stop at their bounds and preserve
all committed progress for the next authorized continuation.

Logical round IDs also enforce the original wall-time and attempt budgets across
platform restarts. An interrupted attempt reserves one uncommitted batch against
its limit; replaying a completed round returns its saved result without new source
requests. This prevents a restart from silently granting another full allowance.

### Continuation recovery: 16 October 2026

Network access recovered and both three-round expansion campaigns were launched
on Modal. The [recovery record](research/continuation-recovery-2026-10-16.json)
contains call IDs, durable campaign claims and a verified baseline: **903 rows,
16,940,394 characters, 2,572,438 whitespace words, 17,019,084 UTF-8 text bytes,
9,712,532 Parquet bytes and 104,038,873 compressed-original bytes**. Both pinned
repository audits passed, including independent checks of the new counters.
These are baseline counts while collection continues; consult HF dataset cards
or `uv run status.py both` for current totals.

### Supervisor recovery

New Modal campaigns persist each child `call_id` before the worker starts loading
its progress. A restarted supervisor can reattach to that same call without
launching a replacement writer. The worker waits briefly for that durable claim;
metadata conflicts use bounded retries with fresh HF state. A dispatch whose call
ID was never saved still stops as ambiguous. Confirm the original child has failed
or stopped before starting a new campaign for that organization.

### Queue a bounded continuation

For a single source, a multi-round campaign can take
`--after-campaign PREVIOUS_CAMPAIGN_ID`. Its supervisor waits read-only for up
to 30 minutes by default for that campaign's `complete` status before claiming new work.
A missing, ambiguous, failed, or timed-out predecessor prevents collection.
Use `--predecessor-wait-seconds` to set a bounded wait up to three hours.
The wait allowance is added to the four-hour campaign deadline; Modal enforces
a seven-hour outer timeout. Worker rounds retain their original bounds. The local
launch record and compute reservation include the dependency; the new HF
campaign plan is written only after the predecessor completes. This is a
finite continuation, not a recurring schedule or a general distributed lock.
Launch only one successor per source.

The [additional collection record](research/continuation-more-2026-10-16.json)
captures the 2,866-document / 182,334,783-character checkpoint and the next
three-round ECB supervisor. EMA's existing three-round campaign continues.

## Sustained expansion: 17 October 2026

The latest adapters share download limits, durable HF storage, raw provenance,
text deduplication, and resumable Modal workers:

| Source argument | HF repository | Initial collection scope |
| --- | --- | --- |
| `ecb` | `jfrick/eu-ecb` | Nine research, bulletin and press collections |
| `ema` | `jfrick/eu-ema` | Six EPAR, news, guidance and committee collections |
| `eurlex` | `jfrick/eu-eurlex` | CELLAR native multilingual XHTML, legislation years 2023/2024 |
| `commission-press` | `jfrick/eu-commission-press` | Original English/French/German Press Corner full text |

`uv run status.py all` includes all mapped organizations. `both` retains
its original ECB/EMA meaning. Launch new sources individually with the same
`modal_cli.py run --detach modal_pipeline.py --source SOURCE --rounds 3`
interface and a unique run ID. Never start a second writer for an active source.

New workers filter completed candidates before weighted fair scheduling.
EMA gives six slots each to assessment reports and product information, and one
to each remaining collection. Assessments currently skew English; product
information preserves multilingual coverage. ECB gives five slots to working
papers, three to occasional papers, and one to each other collection.
These weights prioritize observed text yield without dropping collections or
increasing per-host request rates. They take effect only in newly launched images.

EMA's supported-format check also applies to saved inventories, including valid
Drupal `.pdf-N` filenames. Unsupported containers remain recorded as uncovered
candidates and do not consume download attempts. A newly verified ECB alternate
can be tried once even if the original 404 exhausted its retry budget. Inferred
Monthly Bulletin replacements require matching the inventory title in the first
two PDF pages; filename inference alone is not accepted as identity proof.

CELLAR uses official SPARQL keyset pagination and actual XHTML item URLs, with
5,000-item initial caps per year. Metadata is never stored as document text.
Press Corner uses its official search and JSON document API: only validated
`htmlContent` and official quote fields become text. Raw JSON remains the
stored original, with transformation metadata beside the fetch provenance.
Language fallback and machine translation are rejected at enumeration and
extraction. Its 131,142 search total counts document families/fallbacks, not
131,142 native documents in each language. Initial search scans are bounded to
5,000 rows per language and remain explicitly incomplete.

See [EMA yield](research/ema-yield-20261017.md),
[ECB link recovery](research/ecb-historical-links.md),
[CELLAR verification](research/eurlex-scale-20261017.md), and
[Press Corner verification](research/commission-press-scale-20261017.md).

Transient HF read failures (network errors and HTTP 502/503/504) use at most
three attempts. A transient commit error is accepted as success only when a
fresh pinned head contains exactly the intended progress bytes. Retries retain
the original parent revision; a different writer's state is never overwritten.
HTTP 403/404/409/429 reads are not retried by this storage layer. A commit-only
exception handles HF's explicit “another commit operation is in progress” 409: it
rechecks HEAD and retries only the unchanged-parent write, up to three attempts
with 5/10-second pauses. Ordinary stale-parent conflicts still propagate.

A source can finish its supported inventory while still retaining unsupported or
retry-exhausted candidates. Runs therefore report both `pending_candidates`
(coverage gaps) and `remaining_supported_candidates` (future runnable work).
Campaign completion uses the latter where available, without claiming website
exhaustion.

### Recovery and four active sources

On 17 October, connectivity recovered and all four sources started new
three-round campaigns. CELLAR's preceding campaign completed with 2,997 stored
documents and 133,282,988 characters. Old ECB/EMA worker termination was checked
on Modal and stale HF campaign markers were reconciled before replacements
launched. The [recovery snapshot](research/resumed-four-sources-2026-10-17.json)
records call IDs, pinned repository revisions and 7,592 stored rows across all
sources as collection resumed. This is a timestamped snapshot, not a final
campaign result. New campaigns use weighted scheduling, supported-format
filtering, strict fallback validation and transient HF commit recovery.

### Automatic continuation toward 100,000 documents

The explicit collection goal is configured in `collection_goal.json`. The current
scope counts stored documents from ECB and EMA only, following the owner's
request to focus on these organizations; “EMR” is interpreted as EMA. Existing
EUR-Lex/Commission batches may finish, but the controller allocates no successors
to them. This is not a claim that 100,000 ECB/EMA documents exist.

`modal_goal.py` deploys `lm-exploration-eu-100k`: a single scheduled controller
checks fresh source metrics every minute. It launches bounded source-specific
campaigns, with at most 1,000 attempts and 3,000 work seconds per round, and
one active campaign per source. The eight existing running/queued campaigns were
registered before activation. `goal_pipeline_sources.json` adds five further
CELLAR year inventories for subsequent workers; existing worker images retain
their original configuration. Existing incomplete inventories also expand when
the configured selection limit increases; prior candidates and immutable inventory
artifacts are retained. The attempted limit is saved to avoid repeating an adapter
scan that cannot grow further.

The private dataset [jfrick/eu-corpus-control](https://huggingface.co/datasets/jfrick/eu-corpus-control)
stores claims, Modal call IDs, reservations, observations and source totals in
`state/progress.json`. Corpus Parquet remains in the separate organization repos.
CAS claims prevent two ticks dispatching the same work. An uncertain dispatch
blocks its source for reconciliation. Failed supervisors cannot release their
reservation while a child is still active. Finished reservations settle for one
extra tick so final uploads are included before allocating more work.

No new work is allocated after reaching 100,000 stored rows. There is also a
150-campaign ceiling (including registered campaigns); reaching it is a blocked
budget state, not success. Incomplete discovery, unsupported formats and failed
URLs remain coverage gaps: finishing a saved inventory never means an entire
institution's website has been exhausted. Automatic adapter repair is not
implemented. The first successful live checks observed 8,975 then 8,995 stored
documents; see [activation evidence](research/100k-goal-2026-10-17.json).

```sh
# Inspect current corpus totals; this does not download all Parquet.
uv run status.py all
# Deploy the scheduled controller (requires initialized control state).
uv run python modal_cli.py deploy modal_goal.py
```

To halt future dispatch, stop the `lm-exploration-eu-100k` deployed app in Modal.
Existing campaigns in other apps must be stopped separately if collection must
cease immediately. Reconcile their terminal call trees before resuming; never
clear unresolved claims just to force another launch. All raw artifacts and
local compute reservations remain excluded from Git.

### ECB/EMA expansion and token-goal checks

Future goal workers include EMA's `non_epar_documents` collection with scheduling
weight 2. The [verified inventory and extraction samples](research/ema-extra-collections-20261017.md)
identify 104,799 supported language-level candidates outside the existing document
partitions. The worker selects at most 100,000 of these; candidates and extraction
samples are not counted as stored corpus documents.

Use `uv run status.py both --mapping goal_pipeline_sources.json --cached-tokens` to include the
expanded collection breakdown. The separate billion-token research objective is
not a completed milestone: exact `o200k_base` counts require its vocabulary download,
which the environment proxy still denied on 17 October. Character/byte/word totals
continue to be recorded; estimates must not be used to declare the token goal met.
The owner's subsequent 100,000-document operational limit remains in force.

The future mapping also adds verified ECB annual reports (36 eligible URLs) and
convergence reports (20), each with weight 3. See the
[ECB sample evidence](research/ecb-extra-collections-2026-10-17.md). Annual Report
2025 links 24 language versions not represented by its English inventory; those
links are discovery evidence, not additional downloaded documents.

Deployment of these additions was blocked on 17 October by a Modal API upstream
connection failure, alongside Hugging Face HTTP 503 failures. The last confirmed
stored count was 6,370 ECB/EMA documents at 08:58 UTC. Existing remote workers
were not stopped, but their later state could not be verified. The expanded
mapping is ready locally and must be deployed after connectivity recovers.
The new full-Parquet audit was interrupted during a stalled Xet download and is
not a passed audit. See the [continuation record](research/ecb-ema-expansion-blocked-20261017.json).

### Fresh Modal recovery app

The same harness accepts `EU_MODAL_APP_NAME` and `EU_GOAL_SCHEDULE`. A recovery
run uses the existing HF control state and reservations, so an unresolved or
active source still blocks another writer. Disable its schedule while testing;
creating a new app is not evidence that the existing workers have stopped.

```sh
EU_MODAL_APP_NAME=lm-exploration-eu-recovery-20261017 EU_GOAL_SCHEDULE=0 \
  uv run python modal_cli.py run --detach modal_goal.py
```

The app identity and schedule setting are carried into each function environment so
remote module hydration uses the same settings. The default app and one-minute
schedule remain unchanged when these variables are absent.

Recovery succeeded at 09:37 UTC on 17 October: the fresh app completed a real
controller tick and confirmed 6,889 stored ECB/EMA documents, with both sources
already active. Both the fresh manual app and original scheduled app were deployed
with the expanded mappings. Prior outage records remain historical evidence;
see [successful recovery](research/modal-recovery-success-20261017.json).
The recovery packaging fix passes app settings through function environments,
without adding image build steps after mounted files.

### Hugging Face code backups

Run `uv run backup_code.py` after committing changes. It uploads the committed
`training/data/eu` subtree to the private control repository under
`code/<git-commit>.tar.gz`, with a JSON manifest listing files and the SHA-256
checksum. It then downloads the pinned uploaded archive and verifies its hash.
Only these code paths are written; collection checkpoints and corpus Parquet are
not replaced. Ignored artifacts and uncommitted edits are excluded. This provides
a second code/research backup alongside GitHub, but is not a full Git-history backup.

For an immediate backup attempt followed by retries every ten minutes until a
verified upload succeeds, run `uv run retry_backup.py --interval 600`. The loop
records attempts and the next retry time in ignored
`artifacts/hf-code-backup-retries.jsonl`, then saves the verified upload receipt.
It stops on success or a non-transient error requiring code/configuration repair.


### Public document datasets

The owner authorized public visibility for `jfrick/eu-ecb`, `jfrick/eu-ema`,
`jfrick/eu-eurlex` and `jfrick/eu-commission-press`. Both source mappings now create
future document repositories as public. Existing workers preserve repository
visibility when resuming. The separate `jfrick/eu-corpus-control` repository and
its code backups remain private. Publication does not imply complete website
coverage; source notices and per-document provenance remain in the corpus.


### Annual-report translations and historical convergence reports

The scheduled ECB mapping now follows explicit language-version links on English
annual-report pages, bounded to 100 seed pages. It accepts only links matching
the same report path and identifier, preserving report relationships and discovery
URLs. The 2025 page supplied 23 non-English candidates; live German and French
samples yielded over 360,000 characters each. These probes are discovery evidence,
not additional uploaded-document counts. See the [translation investigation](research/ecb-report-translations-2026-10-17.md).

Three broken convergence-report URLs have exact replacement mappings for 2000,
May 2006 and May 2007. A replacement is accepted only after a primary 404 and a
match against its verified PDF SHA-256. The original inventory identity remains
unchanged. See the [retrieval evidence](research/ecb-convergence-links-20261017.md).
These additions passed the full 195-test suite before deployment; neither implies
complete historical or multilingual coverage.

### Resumable OpenAI token measurement

`token_audit.py` counts `o200k_base` tokens in each stored document's text field,
without adding separators or EOS tokens. It reads a pinned corpus revision,
checks each shard's row count and normalized-record digest, and checkpoints
per-shard results separately under `tokens/<source>.json` in the private control
repository. Subsequent runs reuse verified shard counts. The summary distinguishes
counted rows/shards from total rows/shards and reports completeness only for the
pinned revision; newly ingested documents require another pass.

`modal_token_audit.py` provides a separate bounded CPU audit worker. An hourly coordinator resumes bounded audit pairs. It does not modify collector
progress or corpus data.
The tokenizer loaded successfully on Modal on 17 October; local download remained
unavailable. Counts are actual tokenizer measurements only when the audit produces
a result, never estimates derived from characters.

```sh
uv run python modal_cli.py deploy modal_token_audit.py
uv run python modal_cli.py run --detach modal_token_audit.py --source ecb --max-shards 3000
```

The collection goal also has an explicit `target_tokens: 1000000000`. Each
controller tick sums only cached `o200k_base` counts matching the current corpus
shards by content hash, path and row count. Partial audits provide a conservative
lower bound; missing caches contribute zero and incompatible caches fail the
tick. The latest owner instruction sets `stop_at_token_target: false`, so this
value is a reporting milestone and reaching it does not stop dispatch. Older
configurations default to stopping at the token target. The existing 100,000-document
operational cap and 150-campaign limit remain unchanged. The hourly audit coordinator resumes counting as the corpus grows.

HF may report stale-parent conflicts as either HTTP 409 or 412. Audit checkpoint
retries re-read the destination at the new revision and proceed only if that
source's token-cache bytes have not changed, preserving concurrent controller
updates and rejecting competing audit writes.


### Private visibility restored

The owner subsequently requested that all four document repositories be private
again: `jfrick/eu-ecb`, `jfrick/eu-ema`, `jfrick/eu-eurlex` and
`jfrick/eu-commission-press`. Both source mappings now create private repositories.
Earlier public-visibility records document historical state, not the current
setting. The control repository remains private. Collection and authenticated
resume continue without changing document identities or checkpoints.

Token-audit requests are paced with a two-second pause before each uncached shard,
and checkpoint writes occur every 50 shards plus the final partial checkpoint.
This reduces Hub request and commit bursts. A bounded `resume_after` Modal helper
can defer one ECB/EMA audit pair until a server-requested cooldown expires; its
call ID is recorded before returning. It is not a recurring schedule. HTTP429
responses on 17 October requested approximately one hour, so those retries were
deferred until 15:27 UTC rather than retried immediately.


### Automatic audit continuation

`modal_token_audit.scheduled_tick` runs hourly and reserves at most 48 ECB/EMA
audit pairs, including the pair adopted at deployment. Each worker retains its
1,500-second work budget, 1,800-second hard timeout and shared single-container
limit. Durable reservations and call IDs live in private control state at
`tokens/schedule.json`; active or unresolved calls prevent another pair. A claim
without an attached call ID requires reconciliation instead of replay.

The coordinator respects HTTP429 cooldowns and stops new audit pairs when the
collection controller reports its token/document target or verified exhaustion.
It also stops dispatching at the audit-pair budget. Reaching that budget is an
operational review point, not a claim that the corpus is complete. Both document
collection and audit continuation now run remotely without a local shell session.

### ECB collection expansion after paper inventories

The scheduled mapping adds Financial Stability Reviews (FOEDB type 137), Statistics
Papers (type 5) and Research Bulletins (type 61). Current catalog observations total
224 publication records; these are candidates, not newly stored documents.
English full-text samples from all three series were verified. See
[expansion evidence](research/ecb-next-expansion-20261017.md).
FSR language-picker links are not enumerated: the tested German route contains
the English report plus an unavailable-language notice.

Historical Economic Bulletin boxes from 2014–2018 may try one same-basename
`/pub/pdf/other/` replacement after a primary 404. Only the observed `mb…_focus`,
`eb…_focus`, `eb…_box`, `ebbox…_` and `ecb.ebbox…_` filename forms qualify;
issue year/month or number must match the original path. Every response must be
a PDF of at most 20 pages and contain the full normalized catalog title in its
first two pages. Original inventory identity and both retrieval URLs are retained.
Eight live PDFs passed the integrated rules; this does not validate all remaining
candidates. See [recovery evidence](research/ecb-box-recovery-20261017.md).

### Targeted EMA scientific-history expansion

The scheduled EMA mapping now includes exact EPAR types `variation-report`,
`scientific-discussion` and `scientific-discussion-variation`, each as its own
collection partition. Offline enumeration of the cached official inventory found
4,121 supported document-language URLs: 3,401 variation reports, 529 scientific
discussions and 191 scientific-discussion variations. URL overlap with the prior
configured collections and between these types was zero; organization-wide text
hash deduplication still applies.

The existing extraction pipeline will fetch and validate these documents during
subsequent workers. No new full-text sample or uploaded-row total is claimed by
this inventory-only investigation. Unsupported formats remain explicit gaps.
The broad `epar_documents` umbrella remains excluded to avoid overlapping types
and package-listing-heavy discovery. See [inventory evidence](research/ema-targeted-expansion-20261017.md).

### Raw-storage directory limit and verified recovery

ECB reached Hugging Face's 10,000-files-per-directory limit in the legacy `raw/`
folder. New source blobs now use `raw-v2/<first-two-hash-characters>/<hash>.gz`.
Existing flat originals remain unchanged. Raw hashes resolve through the durable
`raw_blob_metrics` mapping; known originals are reused rather than uploaded under
a second path. Pinned metric backfills support both layouts.

The controller supports an explicit `acknowledge_recovery` operation after a fix
is verified and every source campaign is confirmed terminal. It records the
reason and evidence on the last failed job, preserves all failed statuses, and
resets the prior failure streak. Subsequent failures still trigger the existing
three-failure safeguard. No token, document or campaign limits are increased.

### Read-only matched token status

`status.py --cached-tokens` pins the control revision once and matches cached
`o200k_base` counts against each source checkpoint by shard hash, path and rows.
It reports matched coverage and reservations without downloading Parquet or
starting an audit. An older complete audit summary cannot make a growing source
complete; missing caches are explicit and failed reads do not become zero counts.

### Additional ECB series

The mappings now include Legal Working Papers, Discussion Papers, central-banking
and legal conference proceedings, letters to MEPs and the ECB Blog. See the
[full-text sample evidence](research/ecb-distinct-series-20261018.md). These use
the existing FOEDB adapter and writer reservations; catalog counts are not stored
rows, and proceedings may contain passages also published as speeches.

Historical Monthly Bulletin box proposals now cover 1999–2011 as well as the
previous 2012–2018 range. Thirteen additional live samples passed the unchanged
opening-title and standalone-length checks. The prior 2011 title mismatch remains
rejected by a regression test. See the [year-by-year evidence](research/ecb-box-history-1999-2011-20261018.md).

Four more exact FOEDB types cover Macroprudential Bulletin articles, financial
integration and structure reports, the international role of the euro, and
TARGET annual reports. Each passed a native full-text sample. Their 150 catalog
publication records are discovery opportunities, not saved rows or token counts.
See [sample evidence](research/ecb-report-series-20261018.md).

The exact ECB “Other publication” type (23) adds technical manuals, agreements
and proceedings through `other_publications`. Three selected PDFs passed native
extraction; the inspected catalog contains 772 publication records. See the
[evidence and limitations](research/ecb-other-publications-20261018.md).

Historical article proposals also recognize dotted 1999 Monthly Bulletin page
ranges and guarded 2015–2018 Economic Bulletin article filenames. Six live PDFs
passed the unchanged complete-title validator; 62 remaining article 404s match
the new rules. The unmatched focus-style article is exact text already saved as
a box. See [recovery evidence](research/ecb-article-filename-recovery-20261018.md).

### Additional-document milestone

The owner's earlier 18 October request for another 10,000 documents used the pinned
20,703-row baseline and a **30,703-row target**. Track additional saved rows with:

```bash
HF_HUB_DISABLE_XET=1 uv run status.py both --mapping goal_pipeline_sources.json \
  --cached-tokens --document-target research/additional-10000-documents-20261018.json
```

This read-only option reports growth after the baseline and rejects partial,
wrong-repository or regressed source reads. It does not change collector dispatch,
source pacing or budgets. The owner subsequently superseded this milestone with
collection until documented exhaustion; this command remains historical tracking.
The request is also stored
and verified in the private control repository under its recorded milestone path.


### Prompt EMA continuation and larger batches

The owner requested immediate EMA continuation and larger batches on 18 October.
The existing controller was invoked directly; it reconciled the completed child
tree and started `goal-ema-1792321151-0818560e06` without waiting for the next
scheduled tick. That already running campaign retains its original three rounds.
Future EMA campaigns use `source_rounds: {"ema": 6}` and reserve up to 6,000
attempts. Six sequential 3,000-second rounds provide roughly five hours of work;
ECB retains three rounds. Each worker still allows at most 1,000 attempts and
3,000 work seconds. EMA request spacing remains ten seconds. Increasing the
attempt ceiling alone would not increase throughput: recent EMA rounds reached
the time limit after about 300 attempts.

The controller now runs every minute. When it observes a terminal campaign, it
loads a second fresh source snapshot and reconciles again in the same invocation,
retaining the existing reservation until that fresh read. This removes the
previous extra scheduled settlement interval. Campaign supervisors allow eight
hours; the six-round durable deadline is seven hours, including worker-timeout
headroom. The source mapping explicitly allows six EMA rounds; other mappings
retain a three-round default. Existing jobs and reservations survive the control
configuration migration unchanged. The 100,000-document, 150-campaign and
48-audit-pair limits remain unchanged. The recovery schedule remains disabled.
