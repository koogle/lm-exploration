# Open work

Location: `exploration/latent-workspace/` in the private LM Exploration repository,
main; direct commits and regular pushes authorized. Historical source:
`koogle/latent-workspace`. See [consolidation](../../docs/consolidation.md).

Current priority: [DFlash-style Qwen drafting baseline](docs/dflash-baseline.md).
New authorized work: [larger response-data comparison](docs/dflash-data-scaling.md),
768 versus 6144 nested responses, matched architecture/initialization/exposure and
shared fresh validation. Improved sampling and batched teacher preparation are
implemented. GPU preparation smoke passed (24 training and nine validation responses);
full pool generation is running, then both fits start from scratch. 75 local tests
pass, including fresh training from an archived teacher cache. The existing
reservation ledger was restored to this consolidated checkout; current reservations
are $215.50 after the two preparation workers, not billed usage.
Representation experiments are deferred. One known anchor plus seven masked proposals,
target-generated labels, fixed feature K/V injection, random-anchor packed training
and cached greedy verification are implemented. No useful fast drafter is established.

[Completed fitting study and curves](results/dflash-findings.md): 21000 updates on
380 usable mixed-domain training responses (20661 unique response-token slots),
47 fresh validation responses; 48 test prompts remain ungenerated and unevaluated.
All-response audit: final training NLL 0.0052 and 6.9987 consecutive new matches / 7,
versus validation NLL 9.6334 and 0.2557 matches. The validation-selected checkpoint
at update 17000 reaches only 0.2844 validation matches. Tiny-corpus memorization,
not useful generalization. The same architecture/data were retained throughout.

Actual FP32 verified rollout matched ordinary Qwen on all three measured prompts,
but achieved only 0.879–0.912x AR speed. Three prompts are a pipeline check, not a
benchmark. Original BF16 parity failed due to a reproduced fixed-input argmax flip;
BF16 parity/speed remain unestablished. Training/cached teacher features use BF16.
The selected checkpoint is downloaded and checksum-verified at
`artifacts/dflash-fit-best.pt`; both best/latest checkpoints remain on Modal.

69 local tests pass; actual GPU training, full-response fitting audit and FP32
rollout checks completed. Source/checkpoint/data identities are archived. Failed
initial remote import and BF16 checks remain documented, not relabeled successful.
The original user budget was $200–$500. Current ceiling $500; cumulative conservative
reservations $205.50, not a confirmed bill. No drafting GPU job remains active.

Next: improve target-response coverage while holding the architecture fixed.
The pilot takes first eligible source rows and first-token prose/code prefixes;
a spot check includes license-header continuations and consecutive parliamentary
translation prompts. Use seeded source mixing, varied valid continuation positions,
response-length/repetition audits and more independent validation groups before
larger fitting/data-size runs. Keep existing test groups quarantined. These changes
are pending, not executed. The sufficient corpus size is still unknown; do not
resume layer/J-space experiments based on memorization or decreasing training loss.

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
