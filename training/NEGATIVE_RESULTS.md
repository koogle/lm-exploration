# What didn't work: continuation and Python-focused runs (PRs 25–27, 30)

These four stacked experiments tried to improve the published 124M Simple LLM
(4,999,981,176 targets, 8K context) by continued pretraining or a fresh specialised run. None
produced a model we want to keep. Their pull requests are closed without merging. The
reusable features were split into separate pull requests (listed under each run). One-off
launchers, run-specific code and report builders stay in the branches' Git history. Run
documents and results are in the private HF dataset
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results).
Losses are nats per target on the matched 7,620,187-target, 14-category paired
validation sample (FP32, 2K windows). Repetition means the share of generated
4-grams that repeat. A "long loop" is at least 64 tokens made of at least three
consecutive copies of the same pattern.

## PR 25: one-pass ECB/EMA continuation

- **Goal:** add European Central Bank (ECB) and European Medicines Agency (EMA) domain knowledge to the 5B checkpoint.
- **Setup:** one pass over 878,610,362 ECB/EMA targets on eight H100s, using AdamW
  and the parent's optimizer state.
- **Result:** ECB loss fell from 2.548 to 2.065 and EMA loss from 2.475 to 1.766.
  All 12 original categories got worse, and each 95% paired interval is above zero.
  Python loss rose from 1.559 to 3.994, educational web from 2.675 to 2.911 and
  arXiv from 1.950 to 2.809.
- **Why it failed:** catastrophic forgetting. A domain-only pass with no replay of
  the original mix destroyed general and coding ability.
- **Kept:** single-pass continuation as a per-source `max_passes: 1` cap ([#47](https://github.com/koogle/lm-exploration/pull/47)). Checkpoint
  lineage and lifetime-target counters are part of continued pretraining ([#48](https://github.com/koogle/lm-exploration/pull/48)).
- **Archive:** [`pr-25/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-25)
  (run doc `training/trainer/EU_CONTINUATION.md`, validation report and comparison JSON).

## PR 26: 5B fresh targets with 60% coding

- **Goal:** recover from the EU forgetting and improve coding.
- **Setup:** 5B continuation of the EU checkpoint on fresh data only, with every earlier
  document hash excluded. Coding doubled to 60% and the other sources were scaled by 4/7.
  Run `-r02` completed 4,999,971,081 targets (10.879B lifetime).
- **Result:** Python loss reached 1.421, better than the original 5B model's 1.559.
  The 11 non-coding original sources recovered from the EU parent but stayed slightly
  worse than the original 5B model. ECB lost most of its EU gain (2.494) and so did
  EMA (2.379). Repetition was unsolved: greedy decoding had 81.6% repeated 4-grams
  and long loops in 8 of 10 completions, and T=0.2 had 13 of 20.
- **Why it failed:** better held-out loss did not give usable generations. The model
  still loops under greedy and low-temperature decoding. The first attempt also
  stopped when LibreTexts ran out of fresh data at 5,874,724 targets.
- **Kept:** fresh-only snapshots with a prior-hash exclusion ledger and fresh-capacity
  checks ([#52](https://github.com/koogle/lm-exploration/pull/52)). Also kept: continued pretraining from a checkpoint, which keeps weights,
  the full AdamW or Muon optimizer state and RNG and records lineage ([#48](https://github.com/koogle/lm-exploration/pull/48)), and exact
  integer token budgets ([#47](https://github.com/koogle/lm-exploration/pull/47)).
- **Not kept:** the books-only capacity reallocation (`fresh_book_pool.py`) and the
  60%-coding mix helper, both specific to this run. The matched generation suite
  duplicated the [sampling toolkit](analysis/sampling/README.md), which replaces it.
- **Archive:** [`pr-26/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-26)
  (run doc, dated mixture plan, launch records, validation and repetition report).

## PR 27: fresh multilingual code and gated 10B continuation

- **Goal:** fix repetition and code quality with 10B more targets at 60% code, using
  new permissive multilingual GitHub code at one pass per source.
- **Setup:** CPU collection from pinned `codeparrot/github-code-clean`, a readiness
  audit (normalised dedup and a HumanEval/MBPP 13-gram screen) and parallel shard
  preparation. Training ran on completed databases by reference, with a sampled frozen
  held-out set (the full audit was bypassed at the user's request). The run finished at step 153,342 with
  9,999,963,390 phase targets (20.879B lifetime).
- **Result:** repetition got worse. Greedy repeated 4-grams went from 81.6% to 82.8%
  with 8 of 10 long loops before and after. At T=0.2 the rate went from 77.0% to 84.6%
  and long loops from 13 of 20 to 17 of 20. At T=0.8 the rate went from 23.5% to 25.5%.
- **Why it failed:** 10B more tokens did not reduce repetition, and lower held-out
  loss did not predict better free generation. Simple Python completions were still
  invalid.
- **Kept:** per-source pass caps and capacity checks ([#47](https://github.com/koogle/lm-exploration/pull/47)). Also kept: frozen stratified
  held-out evaluation with per-stratum losses and the `--stop-tokens` evaluation boundary
  ([#49](https://github.com/koogle/lm-exploration/pull/49)), and snapshots that reference completed preparation databases ([#50](https://github.com/koogle/lm-exploration/pull/50)). On the
  data side, multilingual code collection, partitioned preparation with one global
  merge and the read-only readiness audit ([#51](https://github.com/koogle/lm-exploration/pull/51)).
- **Archive:** [`pr-27/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-27)
  (readiness/handoff plan with results, corpus notes and repetition report).

## PR 30: Python-focused 124M / 5B from scratch with Muon

- **Goal:** a fresh 124M model specialised for short Python functions. The mix was
  65% implementation, 15% tests/examples, 10% docs, 5% CS/math, 3% general and 2% books,
  trained for 5B targets with the Muon optimizer from PR 24, now on main.
- **Setup:** private Git snapshots, pinned CodeParrot Python and Stack Overflow Python
  threads. The frozen corpus has 5,949,619,583 unique targets, which is enough for one pass
  per category, and 192 fixed validation windows.
- **Result:** the corpus was frozen and the single-H100 diagnostic passed: tiny-set
  loss went from 10.464919 to 0.001435 in 21 updates, and all 32 MBPP references passed
  the sandbox harness. The eight-H100 qualification was still queued at the last archived
  status (2026-10-21 03:14 UTC). No eight-GPU optimizer update, checkpoint or model is
  recorded.
- **Negative data results:** the repository-only census found only 109,455,272 targets.
  Documentation reached 29,876,984 targets against a 250M minimum before a Stack
  Overflow supplement was added. The Stack and StarCoderData returned `GatedRepoError`.
  The first code preparation failed with a parser `MemoryError`.
- **Kept:** exact per-source target budgets ([#47](https://github.com/koogle/lm-exploration/pull/47)). Also kept: the fixed MBPP tasks and
  sandboxed execution grading of sampling-toolkit outputs ([#53](https://github.com/koogle/lm-exploration/pull/53)).
- **Not kept:** the Python corpus pipeline (`data/python_experiment.py`,
  `python_snapshot.py`, `python_documentation.py`, `python_code_supplement.py`). It
  hard-coded this run's 5B category targets, split salt and source names. Also not kept:
  ordering that prefers collected Git documents, and restart-safe attempt selection
  (`trainer/python_workflow.py`), which served only that run's launcher.
- **Archive:** [`pr-30/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-30)
  (plan, execution record, capacity tables, qualification and harness evidence). The Muon
  and AdamW comparison is under
  [`pr-24/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-24).

## Dropped from Git

Dropped files include the one-off Modal controllers (`modal_eu_*`, `modal_code_*`,
`modal_overnight.py`, `modal_parallel_code.py`, `modal_data_readiness.py`,
`modal_python_*`, `modal_live_publication.py`) and the EU-only training and evaluation code.
Publication of these checkpoints (`trainer/hub_checkpoint.py`, `publication/generate_*`,
`publish_latest_public.py`) is also dropped; use `publication/export.py`. The run-specific
gates are gone too: `workflows/overnight.py`, `data/overnight_snapshot.py`,
`data/original_refresh.py`, and the patch-record ingestion `data/hf_patches.py`. String-built
report scripts under `analysis/` are dropped as well. The run-specific parts of [#40](https://github.com/koogle/lm-exploration/pull/40), the
consolidation PR these features were split from, are dropped as well. Everything remains
in the branch histories of PRs 25–27, 30 and 40. A new run needs a new launcher that follows the Modal
rules in `AGENTS.md`.
