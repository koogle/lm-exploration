# Operations

Use `uv sync --frozen --extra dev --extra modal` and `uv run ...` from
`exploration/latent-workspace/`, not the monorepo root. Relative config, image-build,
artifact and results paths are resolved from that directory.
Run tests, Ruff and diff checks before a verified push. The research branch is
`main`; push it regularly; direct main commits are authorized by the user.

Modal app/volumes are dedicated to this project. Use configured SDK credentials;
do not copy credentials into files or Git. The first remote run must be a bounded
smoke/profile job before larger training.

## Consolidated checkout

The package, lockfile and executable code are preserved unchanged from the source
repository. Keep the existing `latent-workspace-qwen*` app/volume names so historical
checkpoints and caches remain reachable; do not recreate or rename those resources
as part of a GitHub cleanup. No GPU job or deployment is needed for consolidation.

`artifacts/` is ignored by Git. Before a new GPU launch, copy the previous checkout's
`artifacts/budget.json` into this folder's `artifacts/` and reconcile reservations
with actual Modal usage. Do not start a fresh ledger and silently discard earlier
reservations. Download local checkpoints from the existing Modal volume when needed.
The repository migration cannot recover uncommitted files from another machine.

## Managed-cloud connection finding (2026-10-11)

Modal 1.6.1 needs its `api-proxy-support` extra here. grpclib explicitly loads
certifi's bundle, so SSL_CERT_FILE alone did not establish trust for the session
proxy. Appending the environment's trusted public CA certificate to the isolated
Python environment's certifi bundle fixed `modal app list`. Keep TLS verification
and inherited proxy settings enabled. Do not publish the environment-specific CA.

GPU runs, budgets and retrieval commands will be documented once verified.

## Bounded runs

```bash
uv run modal run modal_app.py --config configs/smoke.json --run-id smoke-l40s-01
uv run modal run modal_app.py --config configs/profile.json --run-id profile-l40s --gpu L40S
uv run modal run modal_app.py --config configs/profile.json --run-id profile-h100 --gpu H100
uv run modal volume get latent-workspace-qwen-runs /smoke-l40s-01 artifacts/smoke-l40s-01
```

Each run has a 3600-second remote timeout, no automatic retry, and one GPU.
`max_seconds` stops the training loop earlier and leaves cleanup time. The local
`artifacts/budget.json` ledger reserves the full timeout at conservative allocation
rates ($5/h L40S, $12/h H100). The cap was initially $200; the drafting-first
continuation uses the original user-authorized upper bound of $500. Failed launches retain reservations.
These are allocation assumptions, **not live prices or an account-wide billing
cap**; storage/build costs and runs from other machines are not enforced by this
ledger. Keep the ledger when moving machines and check actual Modal usage. No
long pilot is launched automatically. Resumes need a new reservation/run ID and
`--resume-from ORIGINAL_ID`, with the original config; they extend compute time.

The locked uv environment is built inside Modal. Source is mounted independently,
so code-only changes reuse the dependency image. Results JSON is returned to the
local results/ directory; model heads, optimizer/RNG state, configuration, dataset
hashes and model revisions live on the dedicated runs volume. Model and corpus
caches live on a separate cache volume. Checkpoints exclude frozen Qwen weights.

## Checkpoint recovery and provenance

J-lens calibration saves running matrix sums after each completed prefix. A
resume continues at that prefix count. Decoder checkpoints include AdamW state,
CPU/CUDA RNG states, sampling-generator state, completed update count, lens
matrices and resolved model revision. Resume rejects changed config or source
fingerprints; use the original source checkout to resume a historical run.
The feature cache is per run and config. Interrupted cache writes currently need
manual removal of `features.pt` before resume; completed head/calibration
checkpoints use atomic replacement.

`results/*-manifest.json` and completed run JSON record immutable model/dataset
commits, SHA-256 of the Python package sources, selected layers, seed, calibration
window indices and document hashes. GPU profile results include actual GPU names.
The first smoke predates source-fingerprint reporting and is labeled accordingly.

`uv run python scripts/summarize.py results/RUN.json --output results/RUN.md`
produces a readable evidence table and saved generation samples. A text generation
command for downloaded checkpoints is:

```bash
uv run latent-workspace generate --checkpoint artifacts/RUN/diffusion-multi.pt \
  --prompt 'The capital of France is' --tokens 32
```

Training and generation can use CPU for correctness checks; reported benchmarks
come from Modal GPUs. The tests verify real tiny-Qwen hooks and cached prefix
updates, exact Jacobians against direct autograd, masked-target isolation, frozen
weights, 4/8/16-token sampling, and exact next-update checkpoint restoration.


### Data-study continuation

`study_resume` points to a completed study directory and is normally paired with
`study_parent` pointing there. It restores model, optimizer, sampling and CPU/CUDA
RNG state, verifies critical training configuration and ordered training/held-out
tensors, and carries previous observations forward. It records both source hashes;
unlike the original pilot resume it permits code changes that pass these checks.
An exact two-update versus one-plus-resume regression test covers the CPU path.
Best validation heads are saved separately from resumable final checkpoints.
Use a new run ID for each continuation to preserve the parent's artifacts.

Large raw study reports may be gzip-compressed in results; this is lossless JSON
archival. `scripts/plot_data_scaling.py` reads either JSON or JSON.gz and verifies
matching held-out hashes/groups and critical architecture fields before plotting.


The all-24-layer conditioning runs request 128 GiB host RAM to cache up to three
representation planes (raw residual, attention, J-lens) for 196k windows. They
retain the same bounded L40S reservation/timeout. Cross-backbone corpus reuse is
opt-in and rejects changed source token IDs or base tokenization; it records the
compatibility audit and parent model revision in the report.


### DFlash baseline worker

`uv run modal run modal_dflash.py --config configs/dflash-pilot.json --run-id dflash-pilot-01`
uses a separate short worker (900-second actual timeout, no retries, one L40S).
The same locked ledger reserves $1.25 for its full bound; old 3600-second jobs still
reserve $5. Existing reservations are never reset. The worker shares
the dedicated project volumes and locked uv image. See the baseline protocol for
its data/evaluation gates and the distinction between a smoke run and useful drafting.

The DFlash wrapper defines its image and volumes directly: importing another local
entrypoint module failed in the remote worker because that file was not packaged.
The failed `dflash-pilot-01` app was explicitly stopped; reservation retained.


`modal_dflash.py --long-run` selects an actual 3600-second worker; its reservation
is $5. `configs/dflash-fit.json` resumes the saved baseline/AdamW/RNG on the exact
cached teacher corpus with an explicit new cosine cycle. Cumulative reservations
at launch: $205/$500; the cap follows the upper end of the original authorized
$200–$500 plan. Reservations remain conservative allocations, not billed usage.

Fresh data-size runs can load `teacher_parent` with a balanced `train_limit` while
initializing draft weights and optimizer from scratch. To continue such a run,
set `continue_from` and retain those two fields as checked provenance; restoration
uses the run's archived subset, not a new data selection. As with earlier resumes,
a new warmup/cosine cycle is explicit. The fresh-cache and continuation path is
covered by a tiny-model end-to-end test.

Qwen3.5 quality runs can set `schedule_steps` to preserve one global warmup/cosine
schedule across bounded workers. `steps` is the number of updates requested in
that worker; continuation uses the restored absolute update count for scheduling.
The default zero retains historical new-cycle continuation behavior.

`modal_dflash.py --verify-parent RUN --teacher-audit --run-id AUDIT` uses the
360-second diagnostic worker to check six train/validation source responses for
teacher-token agreement, feature-cache consistency and causal prefix invariance.
It does not train or access test responses.

Full fitting audits now replay the saved training sampler to count conditioning
position and supervised NEW-token exposures separately. Replay must exactly match
the checkpoint sampler RNG state or the audit rejects the count. Conditioning
positions are counted per sampled block; frozen cached features do not imply
new backbone training or target forward computation.

For full fitting audits of the 105M drafter trained for the frozen 2B target, `--verify-parent RUN --fit-audit
--long-run` selects a 3600-second diagnostic worker and reserves $5 for the full
bound. The short worker remains 360 seconds/$0.50; no timeout is under-reserved.

`warm_start_from` loads only a compatible run's selected draft weights onto the
new declared teacher corpus. It starts fresh AdamW, sampler RNG and a new schedule;
it is not optimizer continuation. The report records source checkpoint SHA/step.
A later `continue_from` retains this provenance and restores optimizer/RNG exactly.
CPU integration tests check identical step-zero validation, fresh optimizer step
counts, architecture rejection and continued global-schedule behavior.

`--recache-parent RUN --long-run` adds configured feature depths to existing
train/validation tokens, retaining old feature planes bitwise. Data preparation
with `extend_teacher_parent` retains old cached examples and all held-out prompts,
then generates only uncached training responses. It rejects changed teacher settings
and hash-checks unchanged validation tokens/features. Both use the same bounded
3600-second worker and $5 reservation. No sealed-test teacher generation occurs.

Long DFlash preparation/training and full-audit workers now request 128 GiB host RAM
so expanded feature caches fit without out-of-memory failures. Short smoke/audit
workers remain at 32 GiB. GPU, CPU count, hard 3600-second bound and conservative
$5 reservation are unchanged; these are not billed-cost claims. Existing running
workers retain their original resource snapshot.

For an exposure-study worker that ended at its time bound, download its final
report, then prepare exactly the remaining updates (not another full 20000):

```bash
uv run python scripts/resume_dflash.py results/dflash-exposure-6144-01.json \
  --output configs/dflash-exposure-6144-02.json
uv run modal run --detach modal_dflash.py --config configs/dflash-exposure-6144-02.json \
  --run-id dflash-exposure-6144-02 --long-run
```

The helper rejects live reports, completed fits, unexpected failures, overwritten
configs and invalid schedule budgets. It submits no job. The trainer also rejects
updates past a fixed global schedule, avoiding accidental training at zero LR.
Both sides retain the initial weight hash across workers, separately recording each
worker's starting hash. Use a new run ID/reservation for every continuation.

For the conditional 54k-prompt pool, long workers request 128 GiB host RAM. This
provides room for cached feature tensors plus required-subset validation. The
current training workers keep their original 96 GiB snapshot. The hard timeout,
L40S device, CPU allocation and $5 conservative reservation remain unchanged.


Fresh-grid preparation uses the standalone `modal_long_data.py` app. Its image
and volume declarations are self-contained: importing a second local Modal runner
from the remote entrypoint fails unless that module is explicitly packaged. The
first attempts hit this import failure; no teacher examples were generated, and
failed reservations remain in the ledger. See docs/dflash-fresh-grid.md.


For the fresh grid, local background subprocesses are not durable across session
interruptions: one received a cancellation signal before its first checkpoint.
Use `modal_long_data.py --launch-grid` to spawn the Modal CPU controller. Its handle
is stored in ignored `artifacts/budget_remote_controller.json`; the server run
folder contains `state.json`, `blueprint.json`, and the authoritative `budget.json`.
Download state/ledger with `uv run modal volume get latent-workspace-qwen-runs ...`.
Do not submit local GPU work while the marker exists. Before releasing ownership,
verify the controller and its GPU calls ended, download its complete budget ledger
to `artifacts/budget.json`, then remove the marker. Preserve all failed reservations.
A paused controller's checkpoints and call IDs remain available; budget pause or
training completion does not establish the drafter quality goal.
