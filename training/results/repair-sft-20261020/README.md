# Multi-language repair SFT — first bounded pilot

Parent: PR #26's completed 124.4M / 10.879B checkpoint,
`simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`.
Executable source: `f15d319`; branch `training/coding-sft`, PR #28.

## Verified data

Original procedural arithmetic-repair curriculum, 576 training examples and 36
validation examples across Python, JavaScript, TypeScript, Go, Rust, C, C++, Java
and Bash, with equal example counts per language. Each of the 612 broken/fixed
pairs was compiled/interpreted and tested on Modal CPU: every broken version
failed and every reference repair passed. The frozen tokenizer and train/validation
family/code exclusions passed. Absolute difference and maximum are held out across
all languages; their parameter values also differ from training values.

[Data verification](data-verification.json) records exact dataset and generator
hashes. [CPU run](https://modal.com/apps/radiantai/main/ap-i9TaWXHKdaVgG29SrXeNiM).
This is a small, highly templated synthetic curriculum, not mined repository PRs
or evidence of general software repair ability.

## GPU qualification

[Qualification](gpu-qualification.json) passed on L40S: two uninterrupted updates
match one update followed by a fresh-container resume and the second update,
exactly for weights, optimizer, sampler, CPU/CUDA RNG, counters and identity.
[Qualification run](https://modal.com/apps/radiantai/main/ap-uErJ0M4Wfiuq2KgdalLbQi).

The first attempt failed before training because an imported launcher module was
missing from the remote image. Dependencies were packaged explicitly and the
qualification rerun successfully. The initial local corpus check also caught a
cross-split code overlap; held-out parameters were corrected before verification.

## Pilot

[Training run](https://modal.com/apps/radiantai/main/ap-C4XOrBbWKPfLFVCVvhbinm):
`repair-sft-multilang-20261020-r01`, single L40S, batch 4, LR 1e-5, 200-update
cap, 1,024-token example limit, and 1,200-second loop limit. Full-model AdamW SFT
starts with a fresh optimizer. Only corrected code and EOS are supervised.
Examples are sampled uniformly with replacement. Checkpoints are private on the
`lm-exploration-training` volume; the exact input examples and source/tokenizer
snapshots are stored alongside them.

The pilot completed all **200 updates / 21,531 supervised answer tokens**. Final
checkpoint: `repair-sft-multilang-20261020-r01/step-000000200.pt`.
[Training summary](training-summary.json) retains the baseline, every validation
interval, final per-language loss and training-loss averages.

Training loss averaged 0.09660 over the first 20 updates and 0.0000570 over the last
20. Held-out answer-token loss worsened from **0.86136 to 1.30722**. This is evidence
of fitting the narrow training curriculum without transfer to the held-out
families; it does not justify promoting this checkpoint as an improved coding
model. The parent remains unchanged. No public model upload or merge is performed.

| Language | Base validation loss | Final SFT validation loss |
| --- | ---: | ---: |
| bash | 0.7558 | 1.0341 |
| c | 0.7392 | 1.3696 |
| cpp | 0.7415 | 1.3605 |
| go | 1.2511 | 1.8087 |
| java | 0.7639 | 1.3158 |
| javascript | 0.8649 | 1.4127 |
| python | 0.8949 | 1.3486 |
| rust | 0.7848 | 0.8439 |
| typescript | 0.7470 | 1.1947 |

## Executable generation diagnostic

[Generation summary](generation-summary.json),
[evaluation run](https://modal.com/apps/radiantai/main/ap-r5QquJ9k5JWDWEjf0h7MZc).
Two held-out task families per language, 18 tasks total, greedy BF16 decoding with
128 new tokens and PAD/BOS/UNK suppressed. All nine graders were calibrated on a
reference repair that passes and a broken version that fails before scoring.
Candidates run in separate network-blocked sandboxes; expected outputs remain
outside the sandbox. No code extraction, truncation repair or test rewriting is
used to rescue model output.

| Measurement | Base | SFT step 200 |
| --- | ---: | ---: |
| Correct repairs | 0/18 | 0/18 |
| EOS termination within cap | 0/18 | 18/18 |
| Executed and reached output comparison | 0/18 | 18/18 |

Base generations repeated code and hit the token cap; all failed compilation or
execution. SFT produced bounded executable code, but every candidate returned
incorrect outputs on at least one case. This separates improved formatting and
termination from repair correctness. Raw generations and detailed diagnostics are
stored privately with the pilot checkpoint as `generation-evaluation.json`.

The next data work should broaden verified repair semantics and include real PR
repairs with executable regression tests. The 18-task result is a small diagnostic;
it is not a general coding benchmark, a Qwen comparison, or evidence that more
updates on this narrow curriculum will help. No additional training was launched
after this pilot.

## Validation

13 local CPU tests pass, including frozen-tokenizer multi-language round-trips,
family split checks, known oracle answers, reference Python/shell execution,
answer/EOS alignment, language coverage, length rejection and padding masks.
Ruff and diff checks pass. Actual nine-language CPU verification, exact GPU resume
qualification, completed GPU training and sandbox grader calibration are recorded
above separately from local tests.
