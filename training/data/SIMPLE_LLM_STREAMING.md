# Simple LLM streaming preparation

This implements the first corpus preparation milestone in the
[pretraining plan](../plans/SIMPLE_LLM_PRETRAINING.md). It streams the pinned
12-source recipe from Hugging Face into a persistent token staging database,
exports immutable shards, and exposes resumable PyTorch batches. It does not
launch model training or modify the existing character-level trainer.

## Local bounded preparation

Run from `training/`:

```sh
uv sync --locked --extra corpus
uv run python -m unittest discover -s tests -v
uv run lm-prepare-simple-llm \
  --output runs/simple-llm-stream-v1 \
  --max-records-per-source 100 \
  --export runs/simple-llm-stream-v1/snapshots/pilot-001
```

Repeat with the same output and a **new** snapshot name to continue. The record
limit applies to newly scanned records per source per invocation, including
rejections. `--source code_python` restricts a local preparation call; omitted
sources retain their weights and cannot silently disappear from training.
An empty source in a snapshot makes the loader fail until more data is prepared.
The small record limit is a smoke test, not representative capacity evidence.

Each pinned gzip shard streams over HTTP. Parquet uses HTTP range reads through
the Hugging Face filesystem and small Arrow batches. No whole upstream dataset
download is required. On a disconnect, rerun: gzip replay is sequential up to the
saved row; Parquet can skip preceding row groups. Upstream ordering and pins must
stay fixed. Requests have timeouts; malformed records stop rather than being lost.

The SQLite transaction commits the row cursor, accepted token IDs, exact-dedup
state and filtering counters together. Documents above four million characters
are rejected explicitly, never truncated. Staging has an 8 GiB database bound;
this initial implementation is for bounded preparation/pilot partitions, not a
finished distributed 20B-token corpus builder. One writer owns each version.
Code, recipe or tokenizer changes require a fresh versioned staging directory.
The exported audit file records each scanned source/shard/row and its filtering
decision, tied to the pinned upstream revision.
Tokens and source metadata live in ignored `runs/` locally or the Modal Volume.
Do not commit corpus content or budget ledgers.

## Modal

Install the Modal CLI in the environment configured for the training account.
Restore the existing `training/runs/modal-budget.json` before launching jobs.
The local entrypoint uses the existing conservative reservation function; it
does not initialize a budget or borrow the latent-workspace ledger.

```sh
modal run modal_scripts/modal_prepare_simple_llm.py \
  --version simple-llm-stream-v1 --snapshot pilot-001 --max-records 100
```

The CPU-only worker uses two CPUs, 4 GiB RAM, a 30-minute timeout, no automatic
retries, and the `lm-exploration-mixture-data` Volume. Data lives under
`/data/simple-llm/<version>/`; snapshots under `snapshots/<snapshot>/`.
It commits the Volume after each source and on normal/error exit. A hard timeout
may lose uncommitted Volume changes; replaying the last durable cursor is safe.
Do not start concurrent apps against the same version. Keep failed snapshot
directories for inspection and choose a new name on retry.

## Training interface

Mount the same Volume read-only in the future trainer where practical. Pin one
snapshot for a run; later preparation does not modify that snapshot.

```python
from training_data.token_stream import TokenMixture

# Explicit override is for pipeline smoke tests only, never a substantial run.
data = TokenMixture(
    "/data/simple-llm/v1/snapshots/pilot/manifest.json", context=2048, allow_unreviewed=True
)
x, targets = data.next_batch(batch_size=8, device="cuda")
# logits = model(x), with vocab_size=32768
# loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
# Save this together with model/optimizer state at the same optimizer boundary:
data_state = data.state_dict()
# On resume, construct the same loader and call data.load_state_dict(data_state).
```

Windows are document-local, with explicit BOS input and loss-bearing EOS target.
Long documents use contiguous windows with one token of input/target overlap.
PAD input is 0 and padded targets are -100. Each content token and EOS receives
loss exactly once per pass. Documents are not concatenated across parents.
This sacrifices some packing efficiency for a simple boundary/masking contract.

The scheduler selects the source with the lowest cumulative loss tokens divided
by its weight. Shares converge with a window-sized discrepancy, rather than being
mistaken for document or byte proportions. Actual exposures are saved per source.
Training stops when the next required source exhausts two passes; validation
permits only one pass. It never silently substitutes another source. Immutable
content-hash ordering is deterministic; there is **no random shuffle** in v1 and
thus no hidden shuffle RNG. Checkpoints store manifest/tokenizer identity,
context, split, per-source block counts and loss-token exposures. Schema-2
snapshots also store a seek cursor: pass, shard, document byte offset and token
offset within the document. Resume seeks directly to the next window, without
reading earlier documents or decompressing earlier shards. Failed partial batches
restore these cursors as well as the exposure counts.

## Seekable snapshots and verification

New exports use `uint16-documents-v1`: each `.tokens` shard begins with the eight
bytes `LMTOK01\n`. Each document has a little-endian uint32 content-token count,
32 raw SHA-256 bytes, then that many little-endian uint16 token IDs. The loader
reads at most one window plus its next-token target at a time. BOS/EOS are virtual;
document boundaries, content-hash order, padding and mixture weights are unchanged.
Compressed provenance sidecars retain parent IDs and original metadata without
duplicating the token arrays. The audit remains compressed and outside training's
read path. This trades gzip token compression for bounded, directly seekable reads.

Preparation records checksums and byte sizes once. Training startup hashes only
the manifest and checks shard/audit paths and sizes; it does **not** hash all corpus
or audit bytes. This assumes published snapshots remain immutable. Same-size edits
are not detected at startup. Use explicit verification after transferring a
snapshot or when checking its integrity:

```sh
python -m training_data.token_shards verify /path/to/snapshot/manifest.json
```

Existing gzip snapshots must be converted once into a **new** directory:

```sh
python -m training_data.token_shards convert \
  /path/to/pilot-001/manifest.json /path/to/pilot-001-seekable

# On the existing Modal data volume, without downloading the corpus:
modal run modal_prepare_simple_llm.py \
  --version simple-llm-124m-20261016 --snapshot pilot-001-seekable \
  --convert-from pilot-001
```

Conversion verifies the old checksums, preserves audit status and provenance,
writes the new manifest last, and leaves the original snapshot untouched. It
performs no upstream ingestion or training. The Modal conversion uses the existing
bounded CPU resources and a 30-minute timeout; retain partial outputs for inspection
and choose a new destination if retrying a failed conversion.

Use the converted snapshot for new runs. Existing checkpoints retain their original
manifest/source identities and must resume with their original code and snapshot;
they cannot be resumed against the converted snapshot. No live snapshot conversion
or GPU qualification of this format is implied by the local tests.
The GPU comparison/confirmation launchers now expect `pilot-001-seekable` on the
data volume; run the conversion above before using them with this code.

## Gates before substantial training

Every exported manifest is `training_ready: false`; the loader rejects it by
default. Preparation is not a claim that these gates have passed:

- The original uncommitted collector's exact `mixture-split-v1` hash construction
  is absent. This implementation explicitly uses `simple-llm-parent-v2`; recover
  and reconcile the tokenizer fitting/held-out assignments before relying on its
  evaluation split. It does not claim compatibility with the old split.
- Book chapters group by book identifiers and code files by repository. Paper
  IDs and book identities still need reconciliation across different datasets.
  Exact duplicate contents are retained once; a conflicting split quarantines
  that content entirely in subsequent snapshots. Deduplication is limited to the
  current staging database. Global near deduplication is still required.
- Code licenses are matched case-insensitively to the plan's five SPDX IDs.
  Generated-code flags are filtered. Original metadata, repository/path and the
  pinned dataset revision are preserved. CodeParrot does not necessarily provide
  an original repository commit; a dataset pin is not that missing commit.
  Record-level provenance/eligibility, code quality, credential/PII and benchmark
  contamination audits remain pending.
- Counts are measured only for scanned, accepted records, with content tokens
  plus EOS and separately reported train/validation counts. The two-pass exposure
  ceiling and 20B target comparison are **observations of this snapshot**, not
  upstream capacity estimates. Full-source capacity and final sampling approval
  remain outstanding. Final-test text is not tokenized or exported.

The historical character training loop has since been removed; active training
uses [`trainer/loop.py`](../trainer/loop.py). At this preparation milestone, the planned approximately 150M
model, optimizer integration, GPU profiling and paid execution qualification are
the next milestones; these preparation tools do not imply they are complete.

## Measured local smoke result — 2026-10-16

The [retained report](https://huggingface.co/datasets/jfrick/lm-exploration-results/blob/main/training/results/simple-llm-streaming-smoke-2026-10-16.json)
records a live run against all 12 pinned upstream sources: 32 records each, with
LibreTexts continued to 160 records to reach a training parent. The exported
sample contains 7,066,054 training loss tokens and 999,253 validation loss tokens.
These are bounded sample counts under the provisional split, not usable global
capacity or final evaluation data.

Nine tests passed, including interrupted-stream recovery, exact duplicate split
conflicts, hash guards, token-weighted scheduling, masking and two-pass limits.
A tiny CPU GPT completed an optimizer step using the live all-source snapshot,
and a restored loader produced an identical next batch. Ruff, compilation and
lockfile checks passed. No Modal job launched: volume listing failed, an HTTP
probe returned 503, and the existing training budget ledger was not available
in this checkout. Remote execution remains unverified.
