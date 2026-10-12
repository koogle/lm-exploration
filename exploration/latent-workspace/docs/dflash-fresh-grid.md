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
Sets are balanced and nested by source example identity. Multiple training article
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

Keep the existing 192 validation token/feature sequences unchanged, capped at their
original 64-token trajectories for the primary matched comparison. This measures
improvement on the common evaluation context; it does not establish full 256-token
rollout quality. Add longer-rollout checks when selecting a useful checkpoint.
Report primary validation per domain, position accuracy, NLL, verified rollout
acceptance and output parity. Test stays sealed until the existing quality gate
is met and the selected configuration/checkpoint is frozen. Repeated validation
selection and one seed limit causal/generalization claims.

## Resumable preparation and training

`modal_long_data.py` prepares a pinned large manifest, then bounded teacher chunks.
Each chunk stores a cursor, configuration, manifest SHA and token-only cache;
continuation rejects identity changes and does not regenerate committed examples.
Completed data must have enough usable examples in every domain before a full grid
arm can start. `configs/fresh-grid-*.json` initially contain a deliberately invalid
`/runs/long-data-COMPLETE` placeholder: replace it only with a completed, audited
preparation report, and freeze config hashes before launching.

Local compute reservations remain authoritative; workers are bounded and retain
failed reservations. The former $500 total allocation is still enforced while the
user supplies a replacement spending ceiling for the expanded study. This does not
block preparation within the existing remaining allocation.

References: [DFlash comparison](dflash-fidelity.md),
[quality gate](dflash-improvement-series.md),
[prefix policy](dflash-prefix-coverage.md).
