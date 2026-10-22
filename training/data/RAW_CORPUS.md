# Raw-corpus preparation

The shared [distributed trainer](../distributed/README.md) accepts seekable
token snapshots and `raw-text-v1` SQLite snapshots. Raw snapshots store compressed
document text; `training_data.raw_token_stream.RawTokenMixture` tokenizes document-local
windows as requested. It retains token-weighted scheduling, padding rules, a per-source
pass cap (two by default) and resumable raw-document cursors. Seekable token shards remain the
default prepared format; the two loader cursor schemas are distinct. Select
`loader: "raw"` or `"token_shards"` in the workflow's dataset config. A mismatched
manifest is rejected, never used to switch implementations implicitly.

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

## Explicit small-model workflow

Run commands from `training/`. `workflows/modal_train.py` implements
preparation, qualification and bounded continuation. The selected model config specifies
the modern 9.85M model (eight layers, width 192, three heads).
[`workflows/small-5b.json`](../workflows/small-5b.json) selects
three H100s, a 5B-token target, preparation settings and continuation limits.
The workflow resolves its GPU allocation and token target with the selected model
and dataset configs; non-pilot runs enforce capacity checks. Model dimensions are not
duplicated in Python validation.

The workflow selects `model_configs/modern-10m.json` and
`datasets/scholarly-raw.json`. Model/optimizer settings, dataset selection and
execution settings each have one owner. Use `workflows.modal_train` with
`--workflow workflows/small-5b.json`; selectors can choose another model or dataset.
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
