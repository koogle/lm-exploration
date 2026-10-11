# Open work

Active: user-requested Qwen3.5-2B drafting **quality** comparison. See
[protocol](docs/dflash-qwen35-2b.md). GPU training smoke passed. Raw prompt sample failures prompted a switch to
non-thinking chat formatting; corrected teacher smoke is running. No quality
claim yet. All stopped/failed reservations are retained. Existing 0.5B results below remain the historical baseline.

Location: `exploration/latent-workspace/` in private `koogle/0-to-llm` (LM Exploration),
main; direct commits and regular pushes authorized. Historical source:
`koogle/latent-workspace`. See [consolidation](../../docs/consolidation.md).

Current priority: a useful DFlash-style drafter for frozen Qwen2.5-0.5B.
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
All jobs in this study have completed. Cumulative conservative reservations are
$226.50 under the original authorized $500 upper cap; not a confirmed bill.

Next: establish adequate optimization and data coverage with the same drafting
baseline, guided by held-out accepted multi-token prefixes and verified end-to-end
latency. The 6144-response head's high training loss keeps optimization an open factor.
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
