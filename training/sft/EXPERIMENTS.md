# Repair SFT and CIG-SFT experiments (October 20, 2026)

These runs used the code and launchers in [README.md](README.md). This is a
closed experiment: its run settings are recorded below, not kept as active
recipes. All are single-seed 124M runs.
**No run produced a single correct executable repair.** Lower teacher-forced
loss did not translate into passing code.

**Parent caveat.** Every recorded run except the 20.879B-base repeat started from
`simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`. That is the
124M / 10.879B-token checkpoint of pretraining PR #26, which was not merged. The
repeat started from `simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt`,
the 20.879B-token checkpoint of an unmerged continuation run whose pretraining
included private code.

## Recorded runs

All runs used LR 1e-5, seed 42 and `cig_proximity` 0.3. Steps are the total target;
the CIG long runs resumed each pilot arm's `step-000000200.pt` with `extend`.

| Run | Run ID | Data | GPU | Steps | Batch × context | Eval / save every | `max_seconds` | Objective | Call ID |
| --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| arithmetic pilot | `repair-sft-multilang-20261020-r01` | local curriculum | L40S | 200 | 4 × 1024 | 50 / 50 | 1,200 | sft | not recorded |
| CommitPackFT | `repair-sft-commitpack-20261020-r01` | volume `repair-commitpack-20261020-v1` | H100 | 5,000 | 8 × 2048 | 250 / 250 | 14,400 | sft | `fc-01M5CDKX26XA7DHT7EZEGTE2E3` |
| private patches | `repair-sft-private-patches-20261020-r01` | volume `repair-private-patches-20261020-v1` | H100 | 20,000 | 8 × 2048 | 500 / 1,000 | 14,400 | sft | `fc-01M5CWZ48E3NQRWM3DXXW43BRV` |
| private patches, 20.879B base | `repair-sft-20b-private-patches-20261020-r01` | same volume | H100 | 20,000 | 8 × 2048 | 500 / 1,000 | 14,400 | sft | `fc-01M5DJMJ4S692FC2304JSBPWAA` |
| CIG pilot, SFT arm | `cig-repair-20261020-r01-sft` | local curriculum | L40S | 200 | 4 × 1024 | 50 / 200 | 1,200 | sft | not recorded |
| CIG pilot, CIG arm | `cig-repair-20261020-r01-cig` | local curriculum | L40S | 200 | 4 × 1024 | 50 / 200 | 1,200 | cig | not recorded |
| CIG long, SFT arm | `cig-repair-long-20261020-r01-sft` | local curriculum | L40S | 5,000 | 4 × 1024 | 500 / 1,000 | 3,000 | sft | `fc-01M5CWAE05977DRG7799ACV7HR` |
| CIG long, CIG arm | `cig-repair-long-20261020-r01-cig` | local curriculum | L40S | 5,000 | 4 × 1024 | 500 / 1,000 | 3,000 | cig | `fc-01M5CWAFA32BJPS6DP22CNGHHX` |

The local curriculum is `artifacts/sft/repairs.jsonl` from the curriculum builder.

## Results

Generated outputs are in the private HF dataset `jfrick/lm-exploration-results`:

- [`pr-28/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-28):
  arithmetic pilot and CommitPackFT. Reports, data verification, GPU
  qualification, training/progress summaries, generations.
- [`pr-29/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-29):
  CIG pilot and long continuation. Training metrics, identities, qualification,
  evaluations at each checkpoint, loss plots.

Results from the private-data runs (private patch SFT, the 20.879B-base repeat
and its base-only evaluation, the cancelled full-snapshot preparation) are
**not archived yet**. Their numbers below come from the run records on the
closed PR #28 branch.

Held-out loss is answer-token cross-entropy on each run's own validation split.
Losses from different datasets are not comparable with each other. The executable
diagnostic is the same 18 held-out arithmetic tasks (two families × nine
languages, seven cases each), decoded greedily.

| Run | Data (train / validation) | Updates | Held-out loss | Executable repairs |
| --- | --- | ---: | --- | --- |
| arithmetic pilot | curriculum 576 / 36, 9 languages | 200 | 0.86136 → **1.30722** (worse in all 9 languages) | 0/18 before and after; EOS and runnable 0/18 → 18/18 |
| CommitPackFT | CommitPackFT 37,388 / 561, 9 languages | 5,000 | 0.47947 → 0.34446 | not evaluated |
| private patches | private excerpts + replay 79,441 / 1,617, 12 languages | 20,000 | 0.83449 → 0.56741 | 0/18; 6/18 reach output comparison, 15/18 EOS |
| private patches, 20.879B base | same 79,441 / 1,617 | 20,000 | 0.712214 → 0.521979 | 0/18; 10/18 reach output comparison, 12/18 EOS |

- The 20.879B-token base alone also scores 0/18, with 3/18 reaching output
  comparison and 1/18 EOS. The 10.879B base scores 0/18 with 0/18 EOS. Its best
  validation loss was 0.519240 at step 15,000; the evaluation used step 20,000.
- The pilot overfits: training loss falls from 0.097 to 5.7e-5 (means of the first
  and last 20 updates) while held-out loss rises. SFT teaches the format (EOS,
  runnable code) but not the held-out arithmetic.
- Full private snapshot (`private-patch-full-sources.json`, 1,025 shards, data ID
  `repair-private-full-20261020-v1`): preparation was **cancelled** after shard
  1,020 at 2026-10-20 19:34 UTC. The cause was not established. No dataset report
  was written and no training ran; the planned 40,000-update run
  `repair-sft-private-full-20261020-r01` has no checkpoint.

### CIG-SFT versus ordinary SFT

The CIG pilot arms train both objectives for 200 updates on the arithmetic
curriculum: batch 4, LR 1e-5, seed 42, lambda 0.3, identical answer tokens (21,531).
The CIG long arms extend each arm's own step-200 checkpoint to 5,000 updates
(530,091 answer tokens each). Held-out ordinary cross-entropy starts at 0.861365:

| Step | Ordinary SFT | CIG-SFT |
| ---: | ---: | ---: |
| 200 | 1.307223 | 1.263123 (3.37% lower) |
| 5,000 | 1.590171 | 1.512819 |

- Both arms score **0/18** at steps 200, 1,000, 3,000 and 5,000. All 18 answers
  terminate and run, and the generated code does not change after step 200.
- Both training objectives approach zero: final-100-update medians are 2.3e-7
  (SFT) and 1.8e-7 (CIG). Both arms overfit the tiny curriculum.
- CIG gives positive information to 92.50% of unique training answer tokens, so
  it can reward copying unchanged code rather than the fix.
- The exact GPU extension qualification (steps 201–202, uninterrupted versus
  restored) matched exactly.
- This single-seed 124M pilot gives no support for CIG over ordinary SFT on this
  task. It does not test the paper's larger-model claims.

## Reproduce

From `training/`. To repeat a run, write a recipe in the format of
[README.md](README.md#train) with the settings above in an ignored local path such as
`artifacts/sft/`. The recorded run and data IDs already exist on the volume, so use
fresh ones.

```sh
# Data
modal run -m modal_scripts.modal_sft_data::curriculum --output artifacts/sft/repairs.jsonl
modal run -m modal_scripts.modal_sft_data::commitpack --data-id <commitpack-id> \
  --exclude-dataset artifacts/sft/repairs.jsonl
modal run -m modal_scripts.modal_sft_data::private_patches --sources sft/private-patch-sources.json \
  --data-id <private-id> --reserved <commitpack-id>,<pilot-run-id> --replay <commitpack-id>
# Full snapshot: additionally reserve the earlier private data ID and remove the caps.
modal run --detach -m modal_scripts.modal_sft_data::private_patches \
  --sources sft/private-patch-full-sources.json --data-id <full-id> \
  --reserved <commitpack-id>,<pilot-run-id>,<private-id> --replay <commitpack-id> --full

# Training (example: arithmetic pilot, a workflow with one SFT stage "pilot" from the
# chosen base)
modal run -m workflows.modal_train::main --workflow artifacts/sft/arithmetic-pilot.json \
  --run-id <fresh-prefix> --start pilot --end pilot --qualify 2
modal run -m workflows.modal_train::main --workflow artifacts/sft/arithmetic-pilot.json \
  --run-id <fresh-run-id>
# Volume-backed runs set "data": {"volume": ...}; CIG long runs resume
# <new pilot run>/step-000000200.pt with "extend": true.

# Executable diagnostic
modal run -m modal_scripts.modal_sft_eval --dataset artifacts/sft/repairs.jsonl \
  --checkpoints base=<run>/step-NNNNNNNNN.pt,sft=<fresh-run-id>/step-000000200.pt
```

A pilot's run folder holds a copy of its training examples (`examples.jsonl`), so its
run ID can be reserved like a data ID. The recorded private-patch preparation
reserved the CommitPack data and the arithmetic pilot run, and replayed CommitPack.
The recorded launchers chained the full-snapshot preparation into training
automatically. The generic launchers keep preparation and training as separate
launches. The recorded CIG extension qualification used evaluation every 200
updates. `qualify --steps 202` with a CIG long recipe keeps its cadence;
the cadence does not affect the exactness check.

The private-patch manifests name the two private HF source collections at pinned
revisions. `default_license: MIT` reproduces the license label the recorded runs
gave rows from the MIT collection.
