# Raw-corpus preparation

The shared [distributed trainer](../distributed/README.md) accepts seekable
token snapshots and `raw-text-v1` SQLite snapshots. Raw snapshots store compressed
document text; `training_data.raw_token_stream.RawTokenMixture` tokenizes packed
windows as requested. It retains token-weighted scheduling, padding rules, a per-source
pass cap (two by default) and resumable raw-document cursors. Seekable token shards remain the
default prepared format; the two loader cursor schemas are distinct. Select
`loader: "raw"` or `"token_shards"` in the workflow's dataset config. A mismatched
manifest is rejected, never used to switch implementations implicitly.

Each source is read in insertion order (SQLite row ID), so a cursor is a pass, row ID and
token offset. Preparation inserts each source in upstream shard order, so training sees
documents in that order with no shuffle. Snapshots that reference prepared databases
read each (database, stored source) part in turn.

## Pass caps and exact source targets

A raw manifest source may set `max_passes: 1` to read it at most once (the default and
maximum is two) and `target_tokens` to train exactly that many of its targets. Targets
must be set for every source, fit within each source's pass cap and sum to the run's
`target_tokens`; the scheduler skips a source once its target is met and masks the excess
targets of its last window. When every target is met or a source runs out of passes, the
final partial batch is padded rather than dropped, so every drawn target is trained once.
The trainer and `workflows.modal_train` check capacity against these caps before
allocating GPUs.

Set both when freezing a staging database (from `training/`):

```sh
uv run python -m training_data.export_raw --folder STAGING --destination SNAPSHOT \
  --one-pass code --target-tokens 5000000000
```

`--target-tokens` splits the total by the recipe weights with largest-remainder rounding
(`raw_token_stream.token_targets`), so the per-source targets sum exactly to it. A
single-pass continuation is every source with `--one-pass`.

## Preparation speed and reuse

- **Reuse first.** A finished snapshot stays on the data volume. Train from it with a
  `mode: "snapshot"` dataset config, for example
  [`climbmix-code-v3-2b.json`](../workflows/datasets/climbmix-code-v3-2b.json) for the
  first 2B run's data; no preparation runs. A `prepare_raw` workflow also reuses a
  snapshot whose staging identity matches: the recipe, tokenizer SHA256, split rule and
  `PREPARATION_VERSION` in [`raw_corpus.py`](raw_corpus.py). Raise that version only when
  adapters, filters, deduplication or stored fields change what a recipe prepares;
  other code edits keep prepared data reusable.
- **Whole shards.** `CachedShards` downloads each JSON or Parquet shard to local disk
  before reading it. Downloading a 92 MB ClimbMix shard and parsing it reads about
  36M characters/s, against about 1.6M/s for HTTP range reads.
- **Local staging.** `prepare_and_launch` keeps the SQLite database on local disk (in
  WAL mode) instead of committing every batch to the network volume. A background
  thread copies a consistent snapshot to the volume every `persist_seconds` (workflow
  setting, default 600) while preparation keeps writing; a final copy follows when
  preparation is ready.
- **Overlap.** Each source fetches and tokenizes its next batch while the previous one
  is written. Workflows use 1,024-document batches (still capped at 4 MiB of text).
- **Training during preparation.** With `overlap_preparation: true` in the workflow,
  `prepare_and_launch` copies the database and starts training as soon as every source has
  training data and enough validation tokens for the configured validation windows.
  It writes a `raw-text-growing-v1` manifest whose counts describe that prefix. The
  manifest does not name a database; the loader reads the version's `raw.sqlite` as it
  grows. Preparation then reports in `preparation.json` (training owns `workflow.json`),
  keeps persisting, and writes `complete.json` with the final counts and checked capacity
  when ready. In each training container a [`growing.py`](growing.py) refresher reloads
  the data volume every five minutes and copies each new database to local disk. The
  loader reads the first pass as documents arrive and waits for a newer copy when it
  catches up. Later passes start only after `complete.json` exists. A wait longer than an
  hour fails the run, and the NCCL timeout covers that wait. Exact per-source
  `target_tokens` budgets need final counts, so growing snapshots reject them. Once
  training catches up with preparation, it proceeds at preparation speed; the first step
  no longer waits for the whole corpus. Each local copy is the full database, so the
  training container needs disk space for about two copies.

Measured on one ClimbMix shard in a 4-CPU container (2026-10-24): 759K tokens/s before
these changes, 1.05M with the overlap and 1.17M with 1,024-document batches. The first
2B run's preparation on Modal, before these changes, managed 60–100K tokens/s.

## Explicit small-model workflow

Run commands from `training/`. `workflows/modal_train.py` implements
preparation, qualification and bounded continuation. A workflow whose dataset config
uses `mode: "prepare_raw"` names a corpus recipe; the workflow resolves its GPU
allocation and token target with the selected model and dataset configs, and
non-pilot runs enforce capacity checks. Model dimensions are not duplicated in
Python validation.

Corpus recipes live in [`recipes/`](recipes/); `climbmix-code-v4.json` is the current
one. A source's `files` is an explicit list or a spec that `simple_llm.load_recipe`
expands when the workflow resolves:

- `{"pattern": "shard_{:05d}.parquet", "range": [0, 300, 4]}` formats each index of
  the range (here every fourth ClimbMix shard from 0).
- `{"glob": "raw/*/*.jsonl.gz", "latest_export": true, "order": "interleave"}` lists the
  matching files at the pinned revision, keeps each stream's newest export directory and
  orders them by shard number, then file name. The default order is sorted paths.

Data identity and reuse use the expanded list, which the resolved workflow records.
Resolving a glob needs Hugging Face access. Model/optimizer settings,
dataset selection and execution settings each have one owner.
Qualification uses the selected final training schedule with a short step limit.
Importing or deploying the module does not launch training:
`prepare_and_launch.spawn(...)` explicitly requires approved resources and a fresh
run ID. Run IDs, segment suffixes and snapshot paths are checked before filesystem
access. A staging identity change requires a new data version.

Preparation and loaders check data review status directly.
Reviewed snapshots need no override; pending snapshots require the explicit
`allow_unreviewed_data: true` flag before ingestion or GPU scheduling. New exports
remain `training_ready: false`. No flag clears audits or removes repetition limits.
The obsolete standalone raw-preparation launcher has been removed.

## Prepare once, reuse across runs

Before fetching upstream data, the launcher searches `snapshots/*/manifest.json`
for an immutable snapshot matching the pinned recipe, tokenizer, split/filter
implementation, source weights and two-pass policy. Each source needs validation
data and enough training capacity for the requested mixture. Its database checksum
is verified before reuse; corrupt compatible snapshots fail explicitly.

A match records `data_reused` and uses the same snapshot without redownloading,
refiltering, recounting or copying it. Pending audit flags remain in the manifest.

## Fresh-only snapshots for continued pretraining

[`fresh_raw_corpus.py`](fresh_raw_corpus.py) prepares a snapshot of documents the parent
snapshot never contained. It does three things:

- **Clone:** `initialize_fresh(STAGING, PARENT/manifest.json, [other databases])` clones
  the parent's database, so every upstream cursor continues where the parent stopped. It
  records all prior hashes, held-out ones included, as an exclusion ledger, and returns
  the parent recipe.
- **Prepare:** `FreshRawCorpus(STAGING, recipe, weights).prepare_parallel(target)` takes
  the new phase's mix in `weights`. It runs until each source has enough fresh training
  targets for its share of `target` within two passes. A source whose upstream runs out
  first fails explicitly.
- **Export:** `export(SNAPSHOT, target)` freezes the fresh training documents plus every
  validation document, so evaluation stays comparable with the parent. The manifest
  records the exclusion inputs. Prior documents and conflicts never count as capacity.

## Implementation and measured scope

- `PersistentStream` keeps the upstream connection and decompressor open across
  durable chunks; restart, rollback or shard/source changes reopen from the cursor.
- `CachedShards` retains one compressed shard per active source locally.
- `FastRawCorpus` uses the workflow's configured source workers, tokenizer threads
  and document batch size (currently three, eight and 64). It also bounds each
  batch to roughly 4 MiB.
- One SQLite writer commits documents, decisions, repetition counts and source
  cursors atomically. Deterministic source order, not completion timing, assigns
  duplicate ownership; this can differ from the old whole-source schedule.
- Trigger-maintained token totals avoid repeated scans. Existing staging receives
  one count migration; interrupted downloads retry from committed cursors.

Controlled CPU trials measured 4.3–4.6x faster processing with matched outputs,
excluding upstream downloads, volume commits and count migration. This is not an
end-to-end speedup claim. Producer/consumer overlap is a separate PR #15 experiment.
The [pinned dataset archive](../publication/README.md#pr-13) contains the exact
benchmark reports, completed-run evaluation, samples and negative results.
Historical identifiers and bytes are preserved there; no archived-code loading or
historical-only evaluation entrypoint is maintained in the active package.

## Snapshots that reference prepared databases

[`prepared_reference.py`](prepared_reference.py) builds a snapshot from completed,
immutable `raw.sqlite` databases without copying their text. The snapshot holds an empty
`raw.sqlite`, a manifest with the databases' absolute paths and sizes and the mapping of
their stored source names onto mixture sources, and a frozen held-out set
([`heldout.py`](heldout.py)). The held-out set samples up to 16 validation parents per
stored source, in hash order; train on these snapshots with `heldout: "frozen"` in the
config. `RawTokenMixture` attaches the databases read-only and hides
the sampled held-out documents and every document that shares their parent, across all
databases. Training capacity is counted through that same view and must cover the target
within two passes; a document hash may appear in only one database. There is no global
audit, and the external files are checked by size, not checksum.

From `training/`, where the databases are mounted at the paths training will use:

```sh
uv run python -m training_data.prepared_reference --spec spec.json --destination SNAPSHOT
```

with `spec.json` like `{"weights": {"code": 60, "books": 40}, "target": 10000000000,
"context": 2048, "databases": [{"path": "/data/code/raw.sqlite", "sources": {"code": "code"}}]}`.
Use the training context, so held-out windows fit it.
