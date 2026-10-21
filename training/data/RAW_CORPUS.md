# Raw-corpus preparation

The shared [distributed trainer](../distributed_training/README.md) accepts seekable
token snapshots and `scholarly-raw-v1` SQLite snapshots. Raw snapshots store compressed
document text; `training_data.raw_token_stream.RawTokenMixture` tokenizes document-local
windows as requested. It retains token-weighted scheduling, padding rules, a two-pass
source limit and resumable raw-document cursors. Seekable token shards remain the
default prepared format; the two loader cursor schemas are distinct.

## Explicit small-model workflow

Run commands from `training/`. `distributed_training/modal_small_5b.py` implements
preparation, qualification and bounded continuation for the modern 9.85M model
(eight layers, width 192, three heads) on three H100s, targeting at most 5B tokens
in non-pilot mode. These are dedicated workflow checks in
`distributed_training/run_validation.py`, not shared configuration restrictions.

The active recipes are `distributed_training/recipes/scholarly-modern-10m-5b.json`
and `distributed_training/recipes/scholarly-modern-10m-qualification.json`.
Importing or deploying the module does not launch training:
`prepare_and_launch.spawn(...)` explicitly requires approved resources and a fresh
run ID. Run IDs, segment suffixes and snapshot paths are checked before filesystem
access. A staging identity change requires a new data version.

Preparation and loaders share `training_data.review.require_data_review`.
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

## Implementation and measured scope

- `PersistentStream` keeps the upstream connection and decompressor open across
  durable chunks; restart, rollback or shard/source changes reopen from the cursor.
- `CachedShards` retains one compressed shard per active source locally.
- `FastRawCorpus` fetches/tokenizes across up to three workers, using eight
  tokenizer threads and batches capped at 64 documents or roughly 4 MiB.
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
