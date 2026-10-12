# Fresh-drafter data and response-length study

User direction: train from scratch, expand teacher response coverage and data toward
200k, and compare substantial combinations before concluding that a setting works.
All five runs initialize trainable draft weights randomly with seed 42, fresh AdamW
and sampler. `warm_start_from` and initial `continue_from` are empty. Continuations
are allowed only to complete that fresh run's declared schedule, restoring its own
optimizer and RNG. No external DFlash weights are used.

## Five conditions

| Training examples | Maximum training response tokens |
| ---: | ---: |
| 49152 | 64 |
| 49152 | 256 |
| 98304 | 256 |
| 196608 | 64 |
| 196608 | 256 |

Hold the 79M drafter fixed: five normalized draft layers, target feature depths
2/7/12/17/22, anchor plus seven new proposals, gamma 4, batch 48 and eight anchors.
Each condition initially receives 100000 updates (268.8M NEW-label exposures),
LR 6e-4, 4% warmup/cosine, evaluations every 2000 updates, with intermediate
curves at matched exposure. This is five times the previous screening budget;
it is not a claim that 100000 updates guarantees convergence. Review fitting,
validation and actual exposure before declaring saturation or extending a schedule.
Do not run a separate seed sweep until these substantive comparisons exist.

All arms use the same generated 256-token pool; 64-token arms slice the same
trajectories. EOS can stop earlier: report actual response lengths, tokens, anchors
and distinct article/repository/source-sentence counts, not nominal maxima.
Sets are balanced and nested by source example identity, retaining the prior
49152 training groups with their original token prefixes. All draft weights still
start fresh; group retention permits the two 49152 arms to start from a sealed
subset while remaining teacher preparation continues. Multiple training article
prefixes are correlated contexts and must not be counted as independent documents.
The bounded per-article cap is eight, with article-level splits and historical
holdouts quarantined. Existing teacher prefixes are preserved exactly; only their
nonterminated trajectories are extended. This reuses labels, not learned drafter
weights. Test prompts never enter teacher generation.

## Feature storage and validation

Store training tokens and recompute frozen-Qwen features per sampled batch. This
avoids hundreds of GiB of dense feature storage. All five conditions use this
same path. Validate padding, prefix causality and frozen gradients locally and
with a real-model smoke before full fits. The smoke is a plumbing/profile check,
not one of the five study results.

Extend the same 192 validation trajectories to the 256-token cap, retaining every
original teacher token. This common long view is the primary validation for all
five arms; preserve the original 64-token view for an additional diagnostic. No
validation example is included in training. EOS may terminate early. Report actual
length distributions, per-domain and position accuracy, NLL, verified rollout
acceptance and output parity. Longer-rollout checks are required when selecting a
useful checkpoint. Test stays sealed until the existing quality gate is met and
checkpoint/configuration are frozen. Repeated validation selection and one seed
limit causal/generalization claims.

## Resumable preparation and training

`modal_long_data.py` prepares a pinned large manifest, then bounded teacher chunks.
Each chunk stores a cursor, configuration, manifest SHA and token-only cache;
continuation rejects identity changes and does not regenerate committed examples.
Completed data must have enough usable examples in every domain before a full grid
arm can start. `configs/fresh-grid-*.json` initially contain a deliberately invalid
`/runs/long-data-COMPLETE` placeholder: replace it only with a completed, audited
preparation report, and freeze config hashes before launching.

Local compute reservations remain authoritative; workers are bounded and retain
failed reservations. The user retains the $500 total allocation for now and is open to going further
after reviewing completion costs. The ledger remains fail-closed. This does not
block preparation within the existing remaining allocation.

References: [DFlash comparison](dflash-fidelity.md),
[quality gate](dflash-improvement-series.md),
[prefix policy](dflash-prefix-coverage.md).

Initial online-feature GPU check completed 20/20 updates, three checked verified
outputs match, target gradients remain absent, and initial draft SHA matches the
archived *random initialization*, not a trained checkpoint. `warm_start` is null.
Peak allocated training GPU memory was 8.86 GB on the short-response check.
This is not a fresh-grid quality result or a 256-token memory profile.

`scripts/run_fresh_grid.py` orchestrates bounded preparation chunks and all five
fresh fits, refusing changed package source or initial warm starts. It first extends the existing 54k pool to 256 tokens, then starts the two 49152
arms from that completed pool while the larger pool continues. Required old groups
make those small datasets identical to their eventual nested full-pool subsets. It records live status in ignored
`artifacts/fresh-grid-progress.json`. It does not raise the budget cap or declare
research success when training finishes. Comparison and qualification remain
explicitly pending. Full fits must use identical package source across continuations.

The long-teacher plumbing check generated 96 requested prompts, 95 usable responses,
with prose/code typically at the 256 cap and many translation responses terminating
earlier. Some prose/code teacher continuations are repetitive. Report descriptive
less-repetitive validation slices alongside overall acceptance; neither repetition
nor target-output parity establishes human answer quality. The subsequent 200-update
long-sequence GPU profile completed with all three verified outputs matching and
9.48 GB peak allocated training memory. Its initial SHA again matches the random
initialization and has no warm start. It is a resource check, not a grid quality result.


## Durable Modal controller

The local background runner was canceled with its first GPU input when the session
interrupted. No checkpoint was produced. It is superseded operationally by
`modal_long_data.py --launch-grid`, which spawns a 24-hour CPU controller on Modal.
The submitted controller is `fresh-grid-controller-01`; it starts teacher extension
at worker 02, preserving worker 01's canceled reservation. All GPU workers retain
their 3600-second limits and use the same captured package source. The controller
has a conservative $6 CPU reservation and serializes all new GPU allocations under
the existing $500 cumulative ceiling. It stops new submissions at that cap, allows
current workers to finish, and reports the study incomplete rather than successful.
Server state and ledger live in the controller's run directory on the shared volume.
Local GPU submission is blocked while the ignored ownership marker exists.


If substantial fresh training still fails, the authorized next stage is the
[activation, loss and gradient diagnostic plan](dflash-learning-diagnostics.md).
Current sparse global/projection norms are a starting point; per-layer gradient
flow and projection information loss have not been established. Preserve the
active grid source and use separate checkpoint replays for deeper instrumentation.

## Adaptive stop and accounting reset

User accepted current small-fit quality and prioritized larger data after the
overfitting review. Both 49152-example calls were explicitly canceled with saved
best/latest checkpoints retained: 64 cap at 57593 updates, 256 cap at 37246.
This is an adaptive early-stop deviation from the original equal-100000-update
grid. Compare larger-arm learning curves at common earlier updates; do not claim
a completed equal-final-exposure five-way comparison.

Controller 02 was stopped before the handoff. Controller 04 reattaches the same
active long-teacher preparation call, skips terminal `early stopped` small fits,
and will launch 98304/256 and 196608/{64,256} when the balanced pool is ready.
The training package SHA is unchanged. A local controller regression test checks
that early-stopped fits are not launched again.

At explicit user instruction, cost accounting starts a new $500 epoch. Historical
$468 reservations are archived in controller 02's remote ledger and an ignored
local archive, not erased from provenance. New ledger reserves $12 CPU including the failed coordinator 03 launch; its
future GPU reservations use the new ledger. Already-running teacher preparation
was reserved in the historical epoch. This bookkeeping reset is not a Modal bill
reset. Authoritative ledger and state are now under `/runs/fresh-grid-controller-04/`.

## Start the available intermediate dataset while generation continues

User asks to start using ready prose/code instead of waiting for the entire pool.
A completed teacher chunk already contains more than 32768 usable examples in
each domain, including translation. Seal 98304 examples from immutable chunk 05
using a bounded CPU function; do not read the current worker's changing snapshot
or claim the full pool complete. The subset retains all historical 49152 groups,
unchanged prefixes and identical validation views.

Coordinator 05 starts the fresh 98304/256 fit from this sealed subset while
reattaching existing full-pool generation. This keeps training balanced, rather
than adding an unmeasured prose/code-only curriculum. Both 196608-example arms
retain every group in the sealed 98304 subset, preserving nesting. Draft source
is unchanged; only orchestration gains an available-subset entrypoint. The 98k
arm keeps the same 100000-update schedule, random seed and architecture.

Correction after the available-subset launch: user clarified that the desired
starts were the largest datasets, not the intermediate 98k fit. The 98k call and
coordinator 05 were canceled. Coordinator 06 retains a terminal early-stopped
record for 98k and reattaches unchanged teacher generation; the balanced largest
fits remain scheduled. The option to start prose/code immediately and add
translation is a curriculum change requiring a clear recorded domain schedule,
not relabeling a partial dataset as a complete balanced 196k corpus.

## Authorized large-run curriculum

User explicitly selected: start prose/code now, add translation to the same
models later. Both large-run arms (64/256 response caps) start fresh on a sealed
131072-example pool: 65536 prose and 65536 code. Full validation still covers all
three domains. Training batches contain 24 examples/domain until the transition;
then 16/domain. Training diagnostics cover only present domains and use their
actual macro denominator; validation retains three domains.

The first continuation after full teacher preparation completes switches each
model to the balanced 196608 subset. Keep all initial prose/code examples and the
historical groups, validate unchanged token prefixes and long/short holdout
hashes, restore the same model, AdamW state and RNG, and keep the same global
100000-update cosine schedule. Adding translation does not restart the drafter
or optimizer. The transition is bounded to the next worker boundary, not an
unsafe live cache mutation. Reports record the transition update, domains,
example counts and per-checkpoint stage. This is a curriculum study, not a fresh
all-domain 196k fit from update zero.

The curriculum package changes only opt-in domain selection, subset retention
and explicit dataset transition safeguards. Historical stopped fits keep their
original source SHA. New large fits share frozen source SHA
`47d71073e55a03e87c4bdb59df6aa684cedc532815d30c9f44167ff97da919ae`.
Local checks cover nested two/three-domain subsets, optimizer/RNG restoration,
rejection of undeclared switches and controller transition at worker boundaries.
118 tests and Ruff/format pass. Real-model 20-update check completed with common
random initialization, unchanged holdout hashes, no target gradients, and all
three checked verified outputs matching. Coordinator 07 launched both largest
response-cap arms; no substantial new quality checkpoint exists yet.

Teacher preparation resume compares normalized dataclass configurations so legacy
checkpoints lacking the new default-only training fields remain valid. A CPU
regression resumes a legacy-style checkpoint without regeneration and still
rejects changed response length. The real-model training check captures the
preceding c6578f69 package; its only difference from final 47d71073 is this
teacher-resume configuration normalization, not any drafter/training operation.
Since different response caps can reach worker boundaries at different update
counts, domain-transition timing may differ. Record both; do not interpret their
final comparison as an isolated response-length effect without that caveat.
