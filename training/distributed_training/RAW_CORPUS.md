# Raw-corpus preparation and the completed small-model experiment

The shared [distributed trainer](README.md) accepts seekable token snapshots and
`scholarly-raw-v1` SQLite snapshots. Raw snapshots store compressed document text;
`training_data.raw_token_stream.RawTokenMixture` tokenizes document-local windows
as they are requested. It retains the token-weighted mixture, padding rules,
two-pass source limit and resumable raw-document cursors. Seekable token shards
remain the default prepared format; the two loader cursor schemas are distinct.

## Explicit small-model workflow

`modal_small_5b.py` implements preparation, qualification and bounded continuation
for the modern 9.85M architecture (eight layers, width 192, three heads) on three
H100s, with a non-pilot token target no larger than 5B. These are checks for this
particular workflow in `distributed_training/run_validation.py`, not restrictions
on the shared `DistributedConfig` or on permission to use unreviewed data.

Run project commands from `training/`. The active recipe is
`recipes/scholarly-modern-10m-5b.json`; the qualification recipe is
`recipes/scholarly-modern-10m-qualification.json`. Importing or deploying the
module does not launch training: `prepare_and_launch.spawn(...)` is an explicit
operation requiring approved resources and a fresh run ID. Reuse an existing data
version only when the preparation identity matches.

Preparation and loaders share `training_data.review.require_data_review`.
Reviewed snapshots need no override; pending snapshots require the explicit
`allow_unreviewed_data: true` flag. New exports remain `training_ready: false`.
The workflow rejects missing permission before upstream ingestion or GPU
scheduling. No flag clears pending audits or removes the repetition limit.

## Prepare once, reuse across runs

Before staging or fetching upstream data, the launcher searches the version's
`snapshots/*/manifest.json` for an immutable snapshot with the same pinned recipe,
tokenizer, split/filter implementation, source weights and two-pass policy. Each
source must have validation data and enough training capacity for the requested
mixture. The database checksum is verified before reuse. Corrupt compatible
snapshots fail explicitly rather than being silently replaced.

A match records `data_reused` and uses that exact snapshot without redownloading,
refiltering, recounting or copying its corpus. Changed staging identities require
a new version. Pending audit flags remain in the manifest.

## Preparation implementation and measured scope

- `PersistentStream` keeps the upstream connection and decompressor open across
  durable chunks; restart, rollback, source/shard change or failure reopens from
  the committed cursor.
- `CachedShards` retains one compressed shard per active source locally.
- `FastRawCorpus` fetches/tokenizes across up to three source workers, using eight
  tokenizer threads and batches capped at 64 documents or roughly 4 MiB.
- One SQLite writer commits documents, filtering decisions, repetition counts and
  source cursors atomically. Deterministic source order, not network completion
  order, assigns cross-source duplicate ownership. That ownership can differ from
  the former whole-source-at-a-time schedule for overlapping content.
- Trigger-maintained token totals replace repeated full-corpus scans. Existing
  staging receives one count migration; document format and stored assignments
  remain unchanged. Interrupted downloads retry from committed cursors.

Controlled CPU trials recorded 4.3–4.6x faster preparation with matched stored
outputs. They exclude upstream downloads, volume commits and the count migration;
see `../results/preparation-speedup-20261017.md` and its retained JSON for evidence.
They do not establish live end-to-end speedup. Producer/consumer overlap between
preparation and GPU training belongs to PR #15 and did not produce the completed
small-model checkpoint.

## Historical provenance

The completed run, public model and limitations are described in
`../results/scholarly-modern-10m-5b-completed-20261018.md`. Historical run IDs,
checkpoint formats, HF revisions and result bytes remain unchanged. The package
move, updated source hashes and active config changes mean exact continuation of
old checkpoints requires their original code and settings. New preparation
identities must not be presented as the original frozen corpus identity.
