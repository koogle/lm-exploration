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

## Implemented now

- Ten collection candidates in [sources.json](sources.json), ordered by priority.
- Five source investigations with live HTTP evidence in [research/](research/).
- A bounded collector for HTML/XHTML, selected full-document XML, TXT and
  text-bearing PDF, plus HTML/RSS/Atom/sitemap discovery.
- A repeatable loop over a slice of the registry, raw-response provenance, explicit
  failures, exact-text deduplication and Zstandard-compressed Parquet export.
- Portable per-source agent task briefs, with separate report paths so research
  agents can work in parallel and the parent can integrate their findings.

The [first measured pilot](research/pilot-2026-10-16.md) produced **14 unique
documents, 491,330 characters and 76,499 whitespace words** from four of the five
investigated sources. Commission Press Corner needs a full-text adapter. These
are small-sample observations, not a token forecast.

The research agents in the initial implementation were delegated by the coding
assistant. `pilot.py` itself is a deterministic collection loop: it does not invoke
an LLM or a browser autonomously. Task briefs are inputs to an agent host such as
Codex, not an invented agent API. An unattended agent executor, persistent refresh,
Modal deployment, ASR, OCR, near-deduplication and Hugging Face upload are future
stages. No scheduled or cloud jobs are launched by these commands.

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

uv run pytest -q
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

Every run requires a new directory. Current repeated runs fetch again; ETags and
Last-Modified values are recorded but conditional refresh and resume are not yet
implemented. Generic JSON inventories are not traversed automatically. A source
can therefore be useful even when this first collector yields zero documents.

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

Each run writes:

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
- Maintain a durable source/document registry and raw object store. Use change
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
