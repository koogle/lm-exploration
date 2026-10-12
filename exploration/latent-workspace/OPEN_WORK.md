# Open work

Active: [fresh-drafter data/response grid](docs/dflash-fresh-grid.md). User direction:
train every drafter from scratch and run substantial combinations toward 200k
examples and longer teacher responses. The goal remains **unfinished**.

## Current study

Five fresh fits: 49152/64, 49152/256, 98304/256, 196608/64, 196608/256
(examples / maximum response tokens). Each declares 100000 updates and 268.8M
NEW-label exposures, with identical random draft initialization, fresh optimizer,
fixed 79M five-layer QK-normalized architecture and frozen Qwen2.5-0.5B.
No old learned drafter weights are loaded. Teacher labels/held-out identities are
reused and extended; the primary common validation uses longer trajectories,
with the historical short view retained as an additional diagnostic. Test is sealed.

The 216k training manifest is complete: 72k examples/domain, drawn from 18k prose
articles and 72k source groups each for code and translation. Prose prefixes are
correlated contexts, not independent articles. Usable teacher counts still need
verification after generation/filtering. Prefix safeguards and token-only caching
are integrated into main; the earlier isolated branch is historical.

Real-model online-feature and long-sequence checks completed. Both start from the
archived random initial SHA and have no warm start; all three checked outputs per
profile match the target. The 200-update long profile peaks at 9.48 GB allocated
training memory. These are plumbing/profile checks, not grid quality results.
112 local tests, Ruff/format and diff checks pass.

## Durable execution and budget

Modal controller `fresh-grid-controller-01` is active. It has submitted
`long-teacher-54k-r256-02`, then will start both 49152 fits, prepare the larger
teacher pool and start the remaining three. No full grid fit/result exists yet.

- State: `/runs/fresh-grid-controller-01/state.json`.
- Authoritative ledger: `/runs/fresh-grid-controller-01/budget.json`.
- Local handle/ownership marker: ignored `artifacts/budget_remote_controller.json`.
- Frozen package SHA: `94142e0cee4088a0c5808f0b556a1dcbbd0bb9cba3701cf31c8511aec96c17fa`.

Do not submit local GPU jobs while the marker exists; do not change package source
for active fits/continuations. [Operations](docs/operations.md) describes ledger
sync/release. The server uses bounded 3600-second GPU workers and a bounded 24-hour
CPU coordinator. Current reservations at controller launch: **$397/$500**, including
$6 CPU and retained failed/canceled reservations; this is not actual billed cost.
The user retains $500 for now and is open to reviewing an increase if needed.
Budget exhaustion pauses new jobs and reports the study incomplete.

The first local orchestration/teacher worker was canceled before its checkpoint
when the client session interrupted. `.spawn()` on Modal supersedes that runner;
remote calls now survive local chat/client lifetime. Preserve its failed reservation.

## Evidence and remaining qualification

[Completed fidelity/data findings](results/dflash-improvement-findings.md):
normalization 0.384→0.484 NEW, draft depth →0.520, five feature depths →0.530.
The extra feature-depth acceptance gain is uncertain; all are one-seed results.
Matched data/exposure studies reach 1.022 NEW at 24576 responses and 1.462 at
49152, below the gate. Full earlier exposure audits demonstrate memorization and
large train/validation gaps. Latest expansion audits are no longer running; the
24576 audit has a saved partial report, and no 49152 audit report was saved before
its bound. Do not claim a completed full-corpus audit for that stage.

Remaining for the fresh study:
- Complete teacher preparation; verify actual lengths, source counts, deduplication,
  old-prefix identities, unchanged short holdout and shared long holdout hashes.
- Complete all five fresh schedules; compare curves at equal NEW-label exposure,
  response-length/data interactions, domain scores and position accuracy.
- Report repetition slices; some long teacher continuations are repetitive. Matching
  target proposals does not establish human answer quality.
- Review fitting/validation before further scaling; one seed and adaptive validation
  selection limit significance. Do not confuse 12-response diagnostics with full fits.
- Require the existing macro ≥2 NEW and every-domain ≥1 quality gate, longer verified
  rollouts, then freeze checkpoint/configuration and evaluate sealed test once.
- Source diversity, longer translation tasks, repeated seeds, representation sweeps,
  native pretraining and speed work remain untested/deferred.

Historical [2B-target comparison](results/dflash-qwen35-2b-findings.md) used a
separate 105M drafter and did not improve quality. Hybrid rollout/1.5B size controls
remain unimplemented. Historical [conditioning study](results/conditioning-findings.md)
has an already-used test partition; it is distinct from the fresh sealed drafting
test. Preserve [earlier layer controls](results/small-layer-controls.md) and their
negative findings. See the [knowledge index](docs/knowledge/INDEX.md).

Direct main commits and regular pushes to private `koogle/0-to-llm` remain authorized.
Run project commands from `exploration/latent-workspace/`; use uv.
