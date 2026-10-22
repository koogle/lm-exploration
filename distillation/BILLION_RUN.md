# 215.9M shared-Qwen continuation to 4.5B tokens (paused)

In-progress plan and operations for `modal_billion.py`. Completed-run findings are in
[EXPERIMENTS.md](EXPERIMENTS.md); the full launch, recovery and deployment log is
archived as `pr-31/distillation/BILLION_RUN.md` and `SUPPLY_PIPELINE.md` in
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-31/distillation).

**Status (2026-10-22):** paused. The phased supervisor and its teacher calls were
cancelled before an eight-hour generation window, to qualify Qwen3.5 generation first
([PR #39](https://github.com/koogle/lm-exploration/pull/39)). Data and student checkpoints
are intact. A Qwen3.5 teacher has a different native tokenizer; its targets must stay
separate from this student's shared-Qwen vocabulary.

**Modal rules:** `modal_billion.py` is kept byte-identical to the deployed app
(`lm-distillation-215m-4p5b`). It predates the repository's 24-hour job rule: workers use
`retries=2` and the supervisors respawn themselves across Modal timeouts. Do not launch
or redeploy it as is; a relaunch needs `retries=0`, a persisted 24-hour launch deadline
and a supervisor that cancels outstanding teacher, archive and student calls at that
deadline. It imports `modal_app.image`, `modal_vllm.image`/`volume` and runs
`python -m distillation.scale_train` under torchrun; keep those names when changing them.

## Plan

- Student: the existing 215,910,656-parameter shared-Qwen model, migrated from the
  completed `existing-216m/latest.pt` of the stopped sweep with optimizer, RNG, cursor and
  schedule. Target **4,500,000,000 loss-bearing Qwen targets**, including the first 4.5M,
  from a single pass over freshly generated data. No replay; the final batch is trimmed
  exactly. Settings changes at resume are rejected except the recorded migrations.
- Objective: all-token CE sequence distillation (no probability KL), context 512, BF16
  autocast. LR ramps from about 3e-6 to 3e-5 over 1M tokens, then cosine to 3e-6 at 4.5B
  on the consumed-token count. Validation every 10M, diagnostic generations every 50M.
- Data: the frozen 103,669,090-token prefix of the stopped sweep's corpus, then an
  append-only `manifest.json` of continuation shards from `fresh_producer` teachers
  (Qwen3-8B on one H100 each, prompt indices from 100,000,000). Syntax-checked, not
  execution-verified; exact duplicates are removed per producer only.
- Phases: generate to 1B total accepted tokens, drain teachers, archive every published
  shard to the private dataset `jfrick/python-math-qwen3-distillation-20261021` (exact
  token IDs, immutable commit and checksums in `dataset-archive.json` on the volume),
  then train on one `H100:8` host (eight DDP ranks, 128 global windows per update) up
  to the phase boundary. Repeat at 2B, 3B, 4B and 4.5B. The resume point is the
  61.65M-token checkpoint. `supervise_phased` may request up to 256 teachers; that is a
  configured ceiling, not a measured allocation.

## Operations

From the repository root; the archiver reads the `distillation-huggingface` Modal secret.
Never run two supervisors at once, and stop the previous supervisor and its queued
student before any migration helper.

```sh
modal deploy -m distillation.modal_billion
# One-time migrations, already applied to the current volume state:
#   configure_supply, prepare_unique_policy, configure_phased (.remote() each)
python -c 'import modal; print(modal.Function.from_name("lm-distillation-215m-4p5b", "supervise_phased").spawn().object_id)'
```

Volume `lm-exploration-distillation`, folder
`python-distillation-500m-20261021-01/shared-qwen-215m-4p5b/`: `supervisor.json` (state and
tracked call IDs), `supply-plan.json`, `progress.json` (durable trained targets),
`metrics.jsonl`, `validation.jsonl`, `generations-*.json`, `dataset-archive.json`.
Continuation shards are in `python-distillation-unique-qwen-20261021-01/`. Load the
archived dataset at the recorded commit, not `main`.

Before claiming throughput, measure eight-GPU student tokens/s after at least 5M segment
tokens and median accepted teacher tokens/s; CPU tests cover target accounting, rank
padding and append-only publication but do not qualify an eight-GPU run.
