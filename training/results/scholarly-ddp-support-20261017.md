# Distributed subword training support: verification

This change adds actual CUDA/NCCL training and cross-job checkpoint/resume on
one, two or three GPUs. It builds on the matched benchmark work in PR #10;
the full 20B-token run was not launched as part of this PR.

## Checks

The Modal qualification suite runs 28 unit tests (four configuration tests plus
the existing architecture/embedding/corpus tests). Its CUDA integration uses a
small modern model with the real 32K vocabulary and synthetic documents with
unequal lengths/padding. For each of 1/2/3 GPUs it checks:

- The production accumulation helper's gradients and clipped AdamW update against
  a single-model reference, including entirely masked microbatches.
- Four uninterrupted updates versus two updates, a new torchrun process, restored
  checkpoint, and two further updates. Model/optimizer tensors must agree within
  rtol 1e-5, atol 1e-7; counters, data cursors and all rank RNG states must be exact.
- A token ceiling one token beyond a completed global batch: the next batch is
  rejected and its data cursor is rolled back without exceeding the ceiling.
- Source exhaustion under the two-pass policy, with a final checkpoint and no
  collective deadlock. Two-GPU negative cases also reject incompatible resume
  settings and an unreviewed production manifest on all ranks.

All 28 unit tests and all CUDA integration cases passed in
[the final check app](https://modal.com/apps/radiantai/main/ap-9DhI0yEcBZpsxVMmGKfH98).
[Raw results](scholarly-ddp-checks-20261017-04.json) record maximum parameter
restart differences of 0 on one/two GPUs and 3.7252903e-9 on three GPUs.
All configurations trained the same 925 loss tokens in the four-update fixture.

## Full-model checkpoint and fresh-job resume

The actual 124,389,632-parameter model, context 2,048 and 48-window global batch
ran on **two H100s** through the user-facing launcher:

| Job | Additional updates | Cumulative updates | Cumulative loss tokens | Worker training/report seconds |
| --- | ---: | ---: | ---: | ---: |
| Initial | 2 | 2 | 121,777 | 39.87 |
| Resumed in a fresh Modal app | 2 | 4 | 245,671 | 24.19 |

- [Initial app](https://modal.com/apps/radiantai/main/ap-nydJ75esVp6GAmDAFoeyIi),
  [report](scholarly-ddp-full-smoke-20261017-02.json), checkpoint
  `scholarly-ddp-full-smoke-20261017-02/step-000000002.pt`.
- [Fresh resume app](https://modal.com/apps/radiantai/main/ap-CO0x0CU9w6gLDEdpi1GVka),
  [report](scholarly-ddp-full-resume-20261017-01.json), checkpoint
  `scholarly-ddp-full-resume-20261017-01/step-000000004.pt`.

The resumed initial per-source validation exactly matches the first job's final
validation; token counters continue without reset. Both jobs and the final
integration test record identical hashes for the trainer/config/model/loader/
tokenizer source, checked against the PR files. Both full-model jobs stopped
at the requested two additional updates and committed their checkpoints.
These short timings include validation/save overhead and are not throughput
estimates. Full-model uninterrupted/resumed tensor equivalence is not asserted
by this smoke; the smaller CUDA integration suite makes that comparison.

The CLI timed out waiting for final log streaming on the completed check/resume
apps, but both returned their reports successfully with exit status zero.


## Development failures retained

The initial integration runs
[`ap-rskWOEr1hwdzjba8IulXX4`](https://modal.com/apps/radiantai/main/ap-rskWOEr1hwdzjba8IulXX4)
and [`ap-DQN0UQ0O6Ci9GXcXypT3RQ`](https://modal.com/apps/radiantai/main/ap-DQN0UQ0O6Ci9GXcXypT3RQ)
required bitwise equality of floating checkpoint tensors. One/two-GPU resume
passed; three-GPU resume differed by 3.7253e-9 in the first failing tensor,
consistent with floating-point reduction order across new NCCL processes. The
final test uses explicit FP32 tolerances and preserves exact data/RNG assertions.
This is numerical reproducibility, not a promise of bitwise reproducibility on
all GPU topologies. Logs remain under `scholarly-ddp-checks-20261017-01` and `-02`
on the training volume.

A draft test run `ap-QQBZoDHm1CbMb5BpjkEk7p` was stopped after local Ruff caught an
undefined test variable introduced while changing that comparison. It is fixed
in the final test. Initial full-model launcher app `ap-yf8fP7Yk8hDfGkN57OGSFe`
was stopped while initialization had no workers; the launcher now imports its
local training configuration inside the entrypoints rather than at module import.

## Scope and limits

Checkpoint files are on `lm-exploration-training`, not in Git. The full-model
smoke is a mechanics check with the existing pending-audit pilot corpus, not a
quality result. The diagnostic validation set remains one window/source.
Full-corpus capacity/audits, fast large-corpus cursor recovery, checkpoint storage
retention policy and unattended job orchestration remain outside this change.
See [usage and resume instructions](../trainer/DISTRIBUTED.md).
