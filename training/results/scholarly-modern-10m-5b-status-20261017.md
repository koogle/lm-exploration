# Modern 9.85M / 5B-token experiment

## Completed, validated and published — 2026-10-18

The run completed 4,999,934,057 tokens / 72,805 updates. The final checkpoint,
metric continuity, finite optimizer/parameter state, source-exposure accounting
and both resume boundaries passed validation. The broader evaluation covers
1,798,432 targets from 384 held-out parents: loss 2.6926 versus a matched random
baseline of 10.4369. Samples still show repetition and factual/code errors.

Published: https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b .
See `scholarly-modern-10m-5b-completed-20261018.md` and the retained reports.
The sections below preserve earlier preparation/launch history and are superseded
by this completion status. PR #15 adds future preparation/training overlap; it
was not used for this completed checkpoint.

## Latest: all preparation optimizations deployed

The pipeline now uses batched tokenization on eight tokenizer threads, atomic
64-row writes and three concurrent source workers on 16 requested CPUs. Live
progress confirms Python, FineMath and peS2o are all being collected concurrently.
Code: `4d24240`. Active call: `fc-01M55MH646C21S71HM79JNKNY7`.
The previous call failed on an interrupted upstream gzip response. The replacement
adds cached compressed downloads and bounded retries, preserving committed data.

Controlled Modal CPU benchmarks: 4.27–4.64x faster on synthetic text and 3.57x on
pinned Python/FineMath samples, with identical stored-output fingerprints. These
exclude upstream downloads and volume commits. Live processing handled 8,832
records in a 20.10-second window. See `preparation-speedup-20261017.md` and its
JSON report. This section supersedes the earlier call IDs below.

## Earlier: preparation stream fix deployed

At roughly 651M unique Python training tokens, the preparation worker was stopped
and resumed from its committed corpus using persistent upstream streams. Code
commit: `98dee0b`. New call: `fc-01M55HE3MATB1X0JQPGSBZPNE9`; it replaces
`fc-01M54Q6P2R6P7YH626XRGFVEE5`. Training source hashes and raw-corpus identity
remain unchanged. The launcher now also supports reusing compatible snapshots.
The previous repeated gzip replay at every 1,000 records is eliminated; durable
checkpoints and tokenization policy are preserved. Tests verify identical stored
data and safe cursor reopening. See launch JSON for the current call ID.

## Earlier: automatic 5B workflow launched

The user reiterated the instruction to run the full 5B experiment. It is now
launched as a durable deployed workflow:
https://modal.com/apps/radiantai/main/deployed/lm-exploration-modern-10m-5b .
Run ID: `scholarly-modern-10m-5b-20261017`. Preparation call:
`fc-01M54Q6P2R6P7YH626XRGFVEE5`. Executed code commit: `dff9c86`.

The old preparation app was stopped and its committed corpus/cursors reused.
The deployed workflow automatically finishes capacity preparation, freezes the
raw snapshot, starts the model from scratch under the final 5B schedule, checks
a two-step training + two-step fresh-job resume, and continues on three H100s.
Those real-corpus qualification updates count toward the final target. It
continues through six-hour GPU sessions with atomic checkpoints every 500 updates
and checks each resumed validation boundary. Continuation controllers hand off
automatically, bounded to 20 GPU sessions; no-progress, nonfinite validation,
insufficient source capacity, source exhaustion or unexpected stops are failures.
Completion may fall short of 5B by less than one global batch (98,304 positions),
so the token ceiling is never exceeded. Status is persisted at
`lm-exploration-training/scholarly-modern-10m-5b-20261017/workflow.json`.

`experimental_data: true` is explicitly restricted to the modern 9.85M model
and at most 5B tokens. It permits the authorized experiment with recorded
pending data audits, while preserving the two-pass capacity check, tokenizer/hash
checks, split filtering and exact-duplicate safeguards. It does not change the
manifest to claim completed audits. Pending audits are tokenizer split
reconciliation, global near deduplication, cross-source parent reconciliation,
and eligibility/contamination review. Results remain exploratory.

The new raw loader and deployed GPU worker were qualified on a separate synthetic
fixture: two updates, then two more in a fresh job, totaling 384,270 tokens.
The resumed validation exactly matched the prior final validation. These fixture
tokens are excluded from the 5B experiment. Reports: `raw-10m-gpu-qualify-20261017-01.json`
and `raw-10m-gpu-resume-20261017-01.json`. Config/workflow/raw-loader CPU tests pass.

The workflow was verified running in the **preparing** stage. The full-data GPU
stage has not begun at this status update; it will start automatically. Do not
redeploy changed training source over this active workflow: strict source hashes
protect checkpoint continuation. The sections below retain the earlier handoff
state for context; this section supersedes their manual-launch status.

The user authorized a fresh 5B-token experiment using the smallest modern model
and explicitly retained the planned scholarly/code mixture. The target model has
**9,849,536 parameters**: eight layers, width 192, three heads (64 dimensions per
head), RMSNorm, RoPE, SwiGLU, tied 32,768-entry embeddings, and context 2,048.
It uses the modern model implementation at smaller dimensions. Reusing the old
six-layer width-384 character-model dimensions with the 32K vocabulary would be
23.23M parameters; that is not the selected experiment.

## Completed

- Published the exact 124.4M two-update checkpoint, 121,777 loss tokens:
  https://huggingface.co/jfrick/scholarly-code-modern-gpt-124m-step2 .
  Immutable revision: `a0c833e63ccfe90c00a5b3125cec613f4a301abd`.
  Export checks: exact weights round trip, strict model loading, finite forward,
  parameter count, model source and tokenizer hashes. Public metadata verified
  without authentication. Optimizer/RNG state remains on Modal.
- Qualified the modern 9.85M model on **three H100s**: 20 optimizer steps,
  1,309,624 loss tokens. App:
  https://modal.com/apps/radiantai/main/ap-6RurNo7ZHVtZ3mJdRvTwSy .
- Resumed in a fresh job for two steps, ending at step 22 / 1,434,356 tokens.
  Initial resumed diagnostic validation exactly matched prior final validation.
  App: https://modal.com/apps/radiantai/main/ap-WYf8zxn8Ket1QUfINsF1TF .
  These are pilot-data qualifications, **not progress toward the fresh 5B run**.
  Qualification source commit: `05bbd30`; resume source commit: `8e9fefd`.
  Later raw-loader integration changes trainer hashes; restore these original
  sources to resume those historical qualification checkpoints.
- CPU tests cover raw-text preservation, durable restart, duplicate split
  conflicts, frozen snapshot export, inside-document loader resume, partial-batch
  rollback, and exact two-pass exposure accounting. Existing corpus/config tests
  also pass (17 tests total).

## Active preparation, not full training

CPU app: https://modal.com/apps/radiantai/main/ap-Xv3cWev6KOA1D63FfBGZGy .
Four CPUs, 8 GiB memory, maximum 24 hours, no retries. The first attempt
`ap-iHvqM2UnBines9FFCWyaMj` failed to import its launcher dependency and was stopped;
the dependency was packaged before this replacement was started.

Data volume: `lm-exploration-mixture-data`.
Path: `scholarly-code/scholarly-small-5b-raw-20261017/`.
Durable status: `capacity.json`; staging database: `raw.sqlite`.
It collects pinned raw documents, source metadata and split assignments, filters
licenses/special-token collisions, removes exact duplicates, quarantines detected
cross-split duplicates, and measures nonpadding token capacity. Tokenization for
capacity measurement is transient: it saves compressed text and counts, not token
ID shards. Test text is excluded from stored text. Every bounded chunk commits
its row cursor and database to the volume. At most 64 GiB SQLite storage.

For 5B exposures with at most two passes, minimum unique train tokens are:

| Source | Share | Minimum unique train tokens |
|---|---:|---:|
| FineWeb-Edu | 10% | 250M |
| Python code | 30% | 750M |
| FineMath | 20% | 500M |
| peS2o | 10% | 250M |
| PubMed | 3% | 75M |
| arXiv | 2% | 50M |
| Pressbooks | 3% | 75M |
| LibreTexts | 3% | 75M |
| DOAB | 4% | 100M |
| Gutenberg | 5% | 125M |
| Pre-1929 books | 7% | 175M |
| Library of Congress | 3% | 75M |

The worker aims for 5% headroom and reports upstream exhaustion rather than
silently changing weights or repeating the pilot. Later duplicate conflicts can
reduce earlier sources' counts; final capacity must be checked again.

**The 5B-token training job has not started and is not scheduled to start
automatically.** Preparation remains pending capacity and corpus qualification:
tokenizer split reconciliation, cross-source parent/near-duplicate checks,
eligibility and contamination review. The existing pilot snapshot cannot support
5B tokens and has pending audits. Export retains `training_ready: false`; it does
not manufacture approval by changing that flag.

## Training support

The 5B recipe is `recipes/scholarly-modern-10m-5b.json`. Fresh initialization,
three H100s, 48 global windows, microbatch eight per GPU with accumulation,
20M-token warmup, AdamW peak LR 3e-4, cosine decay to 10%, checkpoint/diagnostic
validation every 500 steps. This is an initial recipe, not a tuned optimum.

After preparation stops, `python -m training_data.export_raw` freezes a SQLite
snapshot and checksum. Raw snapshots use `scholarly-raw-v1`. The distributed
trainer dispatches to `RawTokenMixture`, selecting a source by token deficit and
tokenizing one document at a time to fill windows. Leftovers persist in the
generator; checkpoints record document hash, pass and token offset. Resume seeks
to the document instead of retokenizing the entire earlier stream. Documents stay
separate, with BOS/EOS and ignored padding. Cross-document packing and asynchronous
prefetch are not implemented. The raw path has CPU tests but still needs an
end-to-end GPU qualification on the frozen snapshot before the full launch.

The launcher now allows an explicitly requested `--max-seconds 82800` (23 hours)
within a 24-hour Modal worker timeout, leaving time to checkpoint and exit.
The default session remains 3,000 seconds. No automatic continuation is implied.

Inspect capacity after a committed chunk:

```sh
modal volume get lm-exploration-mixture-data \
  scholarly-code/scholarly-small-5b-raw-20261017/capacity.json \
  runs/small-5b-capacity.json
```

The previous 20-step result is too short to promise a sustained 5B runtime,
especially with the new raw-text loader. Measure it on the qualified corpus.
