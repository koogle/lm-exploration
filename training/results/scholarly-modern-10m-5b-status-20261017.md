# Modern 9.85M / 5B-token experiment

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
