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
