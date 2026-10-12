# Open work

Active: [small-target drafter improvement series](docs/dflash-improvement-series.md).
All matched fidelity fits and full-training audits completed. Selected validation NEW:
original 0.384, normalization 0.484, five draft layers 0.520, five feature depths
0.530. All 12 checked rollouts match per new fit. Extra feature-depth benefit is
uncertain; full-training fitting remains weak. The selected 79M checkpoint is
`dflash-fidelity-features-01/best.pt`, step 18000. Frozen target: Qwen2.5-0.5B.

**Current (2026-10-12):** Both matched exposure fits completed 20000 updates and
53.76M additional NEW-label exposures. Selected validation NEW matches / 7 are
**0.57378 (6144 responses) versus 1.02179 (24576)**. The paired conditional bootstrap
95% interval for the gain is [0.37974, 0.51880]; this excludes selection bias and
seed variation. Both pass all twelve checked FP32 rollouts. The two-token gate is
unmet. See `results/dflash-exposure-scaling.png` and its JSON evidence sidecar.

The small arm's continuation `dflash-exposure-6144-02` restored optimizer/RNG and
completed the remaining 898 updates on the original global schedule. Reports share
initial-weight, source and validation hashes. The smaller full-training/zero-memory audit `dflash-exposure-6144-audit-01`
is complete: final training NEW 6.999959 versus validation 0.44514; selected
training NEW 2.30198 versus validation 0.57378. Validation entries and sampler
replay match exactly. `dflash-exposure-24576-audit-01` is also complete.
The larger selected checkpoint has now been fully audited: training NEW 3.19508
versus validation 1.02179, with exact validation agreement and verified downloaded
checkpoint SHA. Final larger-corpus training NEW is 5.28722 versus validation 0.95882; both
audit validation dictionaries and sampler replay match the training reports.
See `results/dflash-exposure-fit-summary.json` and `dflash-exposure-full-fit.png`.

The conditional 54k-prompt pool has 53476 usable training responses (17769 prose,
17898 code, 17809 translation), preserving the five-depth parent cache and all 192
validation examples. Both selected audits informed the next matched pair, now running as
`dflash-expansion-24576-01` and `dflash-expansion-49152-01`, from configs
`configs/dflash-expansion-{24576,49152}.json`. Both use the same selected 79M weights
(`/runs/dflash-exposure-24576-01/best.pt`, step 10000), fresh AdamW/sampler, batch48,
eight anchors, 6e-4 LR and one fixed 20000-update schedule. Keep package source
unchanged; resume only remaining updates after bounded workers if needed.
Initial worker allocation is four $5 reservations for this pair. Launch checks pass: identical initial weights/source/validation hash and exact
step-zero validation matching the selected source head. Both report all 24576
required training groups after the runtime token-identity checks. See
`results/dflash-expansion-launch-audit.json` and selection provenance.
No test responses have been generated; keep test sealed until the declared gate passes.
Interim curves/snapshots: `results/dflash-expansion-interim.{png,json}` (5500
control updates, 4500 larger-data updates). These are incomplete; compare matching
update positions and do not infer full-corpus fitting from twelve train diagnostics.
Before any further large pool: `results/dflash-prose-source-capacity.json` counts
23090 eligible training article groups before filtering, so the current one-prompt
per-group recipe has a balanced upper bound of 69270 responses. A 98304-response
request needs a revised, validated source/prefix policy; do not blindly increase
the count. Current runs are unchanged. See the protocol for deferred options.
An isolated prototype is pushed on `drafter-data-coverage` (`0d1a68c`), checked
out at `/workspace/drafter-data-coverage`. It adds training-only article-prefix
primitives/registry guards and private mapped teacher-cache reads; 100 CPU tests
pass. It is NOT merged, not wired into pool preparation, and has not run on GPU.
Keep main package source unchanged while the active fits/resumes run. Before using
the branch, follow its `docs/dflash-prefix-coverage.md` integration/smoke checklist.
The budget ledger in this main checkout remains authoritative.
Goal is NOT complete; representation experiments and speed work remain deferred.

82 local tests pass, as do Ruff/format/diff checks. Conservative reservations are
**$348.50/$500**, not billed cost. Retain failed/stopped reservations. Stop the
active goal only when useful quality is demonstrated or further work is blocked.

Historical completed: [Qwen3.5-2B quality comparison](results/dflash-qwen35-2b-findings.md).
Both targets used 6144 responses and 20000 updates. Selected shared-prompt NEW
matches are 0.170 (2B) versus 0.387 (0.5B), worse in each domain. Full-training
2B selected/final matches are 0.173/0.175 versus validation 0.170/0.168. This is
poor fitting as well as poor held-out prediction, not the tiny-set memorization
pattern. Family, tokenizer, prompt format and draft capacity differ; no pure-size
or human answer-quality claim. Test remains sealed and hybrid rollout unimplemented.

Both full fitting audits match the training curves exactly. Sampler replay confirms
3.36M NEW-label exposures and 72.97M conditioning-position exposures. The downloaded
selected checkpoint matches the GPU hash. 79 local tests and all GPU checks passed;
all 2B comparison jobs ended. Reservations at that historical completion: $258.25/$500,
not billed usage; failed/stopped reservations are retained.

Current follow-up is the small-Qwen series above. The 1.5B size control and hybrid
adapter remain untested and deferred. The [reference audit](docs/dflash-fidelity.md)
identified missing QK normalization; the new matched 0.5B-target study measures an
improvement, with weak larger-set fitting still unresolved. Representation sweeps
and inference speed optimization remain deferred.

Location: `exploration/latent-workspace/` in private `koogle/0-to-llm` (LM Exploration),
main; direct commits and regular pushes authorized. Historical source:
`koogle/latent-workspace`. See [consolidation](../../docs/consolidation.md).

Current priority: useful block prediction with a frozen Qwen target.
Representation/layer/J-space experiments remain deferred.

[Completed data-size study](results/dflash-data-scaling.md): fresh 768- and 6144-response
fits, same initial weights, architecture, 20000 updates, 3.36M repeated new-token
label exposures and 192 shared validation responses. All identity checks match.
The common pool contains 6538 usable training responses, generated in batches from
shuffled sources with varied prose/code prefix offsets. Source revisions, groups,
near-duplicate screening, dropped rows, actual counts and hashes are archived.
New 192 test prompts have no generated labels or metrics; earlier holdouts remain
quarantined. This study is within-source generalization, not arbitrary-domain transfer.

Best validation NEW matches / 7: **0.2185 -> 0.3844**. The paired conditional bootstrap
interval for the gain is [0.1343, 0.1993]; it does not capture selection bias or seed
variation. All three validation domains improve. At the final update, small-set
full-response train/validation NLL is 0.1377/9.3788; large-set is 5.9707/6.3464.
More data reduces overfitting, but the larger head is still weak on training as well
as validation. This does not identify a sufficient dataset size or rule out further
optimization/capacity limitations. Both full-response audits evaluate up to eight
fixed anchors/response, not every possible anchor.

All twelve shared FP32 rollout prompts matched Qwen for both heads. Median prompt
speed ratios are 0.994x and 1.095x; this small check includes a repetitive-zero code
outlier and is not a broad speed qualification. BF16 output parity remains unresolved.
Training and cached teacher features use BF16; correctness-reference rollout is FP32
on upcast stored weights. Speculation preserves target behavior, not human answer quality.

75 local tests pass. Both actual GPU fits, full-response audits and rollout checks
completed. Selected heads are downloaded and checksum-verified at
`artifacts/dflash-scale-{768,6144}-best.pt`; best/latest checkpoints persist on Modal.
Both selected heads also ran through the CLI in the consolidated checkout.
All jobs in that study completed. At that checkpoint, conservative reservations
were $226.50/$500; the current cumulative reservation appears above.

Next: establish adequate optimization and data coverage with the same drafting
baseline, guided by held-out multi-token prediction quality. The 6144-response head's high training loss keeps optimization an open factor.
Further source diversity, longer teacher responses and larger mixed-domain pools
remain untested. Bounded streaming shuffle is not globally uniform sampling, Python
is the only code language, and translation separation is sentence-level. Keep test
sealed while making these decisions. Do not resume representation sweeps or claim
that scaling data alone is guaranteed to solve useful drafting.

Earlier DFlash work: [380-response fit](results/dflash-findings.md) demonstrated
memorization after 21000 updates but only 0.284 best validation new matches on a
different smaller holdout. Its sampling and validation differ from the new study;
it is historical evidence, not the matched small-data control. The original BF16
verifier failure and fixed-input arithmetic diagnosis remain archived.

The [trained conditioning study is complete](results/conditioning-findings.md).
Executed: 96 residual/attention probes across two seeds; nine Qwen2.5-0.5B decoder
conditions and two Qwen3-0.6B conditions at 4k updates each; one frozen-checkpoint
test suite across all 11 heads. Every decoder used the same 196,608 mixed-domain
training windows. Full corpus identity checks and source hashes are retained.

Findings: all-layer residual conditioning improves test NLL over final-only for
both Qwen families. The compressed J-lens loses to matching raw states. Diffusion
does not beat the parallel baseline here. The all-layer Qwen2.5 diffusion head
matches only 1.141 consecutive Qwen tokens / 4, with 1.6% full-block agreement.
Absolute generation quality remains poor. The test split has now been evaluated;
it must not be described as sealed or untouched for subsequent adaptive work.

At completion of the original study, 50 local tests and both real-model GPU
smokes passed, with $192/$200 reserved (not actual billed usage). The
all-layer inference checkpoint is downloaded and checksum-verified against the
tested head; other best/final checkpoints remain in the dedicated Modal volume.

Remaining:
- Improve later-token generalization using controlled training/decoder changes;
  human corpus labels were used, not Qwen distillation targets.
- Larger fresh validation/test partitions and near-duplicate audit before further
  adaptive experiments. Current slices are only 64 windows/domain, Python code,
  WikiText prose and English-to-German translation.
- Repeat trained decoder comparisons across seeds; only the probes and earlier
  dataset-size comparisons have repeated seeds so far.
- Qwen3.5-0.8B requires a separate tokenizer/data preparation path and real-model
  adapter verification. It was not trained in this study.
- More J-lens calibration/convergence and readout controls before broader claims
  about the paper's J-space representation.
- No collaborative module or native backbone pretraining in this phase.

Follow-up complete: [small-data layer controls](results/small-layer-controls.md).
All 12 fits (96/768/3072 windows × baseline/dropout/null/both) reached 3200 updates,
batch 32, on one bounded L40S job. Baseline final training accuracy:
100%/98.9%/94.3%; every size overfits validation. At 3072 windows, best validation
NLL is 7.264 baseline, 7.333 dropout, 7.262 null, 7.314 both. No clear gain from
these controls; one seed and a reused small validation slice are exploratory.
Tiny-set fitting works, but this does not establish enough data for a useful drafter
or convergence at 3072 windows. Retain larger-data work as the generalization
baseline; do not replace it with the tiny memorization sets.

At the small-control checkpoint, 55 local tests passed, including exact dropout
resume and compact feature caching.
All four controls completed actual frozen-Qwen GPU training. Source, ordered
subsets, split hashes, supervision counts and full-training diagnostics verified.
That completed checkpoint had $197 reserved against the initial $200 allocation.
See the current drafting-first status above for subsequent reservations and cap.
Checkpoints: `/runs/small-layer-controls-01/{baseline,dropout,null,dropout-null}/`.
