# Decisions

- 2026-10-11 (consolidation): Move this self-contained project into
  `exploration/latent-workspace/` in LM Exploration. Preserve its Python package,
  lockfile, experiments, measured evidence and Modal resource names. Run its
  commands from this folder. The separate `training/` effort covers from-scratch
  models and a future Modal Run Board; frozen-backbone restrictions here do not
  apply there. The independent-repository decision below is historical.

- 2026-10-11: Use frozen pretrained Qwen2.5-0.5B. Native pretraining and collaborative
  generation are outside phase one per user clarification.
- 2026-10-11: Compare the same four-token block decoder across J-lens readout,
  individual residual/attention layers and a configurable selected-layer set.
- 2026-10-11: Independent private repository, direct main commits, regular pushes,
  uv dependency management, and an AGENTS.md documentation workflow adapted from
  Age of Agents, as requested by the user.
- 2026-10-11: User requires distinct train/validation/test data and separate code,
  translation and prose reporting. Extend the exploration to pretrained Qwen
  checkpoints through approximately 1B parameters; do not treat cross-family
  comparisons as pure parameter-count scaling.
- 2026-10-11: Train one shared drafter on all three domains; domain categories
  define evaluation slices, not isolated domain training. Keep test sealed during
  budget selection. Prioritize data-size/optimization learning curves before
  the expanded conditioning and model-size sweep.
- 2026-10-11: After three-seed sizing and a 16k-update continuation, retain
  196,608 mixed-domain windows and a 4k-update/batch-128 screening budget for
  decoder diagnostics. Larger data consistently helps, but poor later-token
  generation and diminishing optimization returns do not justify further data
  expansion yet. This is a diagnostic choice, not sufficient-quality evidence.
- 2026-10-11: Proceed with the requested trained layer/J-lens conditioning study
  now, using the fixed diagnostic budget; do not defer the central experiment
  until generation quality is solved. Keep probe and decoder conclusions separate.
- 2026-10-11: Completed nine Qwen2.5 and two Qwen3-0.6B trained conditions and
  evaluated all validation-selected heads on test once under a committed hash-
  checked protocol. Retain all-layer residual conditioning as the strongest
  tested diffusion memory, but do not claim a diffusion advantage or useful
  generation. Use fresh held-out data for subsequent adaptive experiments.
- 2026-10-11: Follow the fixed-budget study with dense small-data convergence
  diagnostics and all-layer dropout/null-attention controls. Retain the existing
  validation slice for exploration only; do not adapt on another test evaluation.
  See [protocol](docs/small-data-controls.md).
- 2026-10-11: Completed the small-data controls without promoting a new decoder:
  zero-content attention is effectively tied and 25% layer dropout gives no clear
  improvement. Tiny sets are fitting diagnostics, not replacements for the larger
  mixed-domain training pool. Preserve this negative result and the full curves.

- 2026-10-11: User redirected the project to a DFlash-style drafting baseline before
  further hidden-representation experiments. Use target-generated responses,
  known-anchor blocks, one-pass prediction, and actual target verification. Count
  only accepted new proposals, not the anchor. Keep fixed standard feature context;
  defer workspace/J-space/layer ablations. Protocol: docs/dflash-baseline.md.
- 2026-10-11: Add a genuinely shorter 900-second worker and reserve its full bound
  at the existing $5/hour rate ($1.25). Preserve the $200 cap and all old reservations;
  do not shorten a reservation while leaving a longer worker timeout in place.

- 2026-10-11: Initial DFlash-style training remains underfit. Continue the SAME
  baseline on its exact cached target-response corpus before architectural work.
  Restore AdamW/RNG and declare a new cosine schedule; measure fitting and acceptance.
- 2026-10-11: Honor the original user-authorized $200–$500 allocation range by moving
  from the initial conservative $200 ledger ceiling to its $500 upper bound. Retain
  all old reservations; reserve only one additional bounded $5 training job now.
- 2026-10-11: BF16 GPU AR/block parity failed due to a reproduced fixed-input target
  argmax flip. Use FP32 arithmetic as the correctness reference; retain the failure,
  do not label the initial BF16 rollout successful or claim a validated speedup.

- 2026-10-11: Completed the same-baseline DFlash fitting study through 21000 updates.
  Full-response training fits nearly perfectly; held-out acceptance stays below
  0.3 new tokens / 7. FP32 verified output matches three prompts but is slower than
  AR. Prioritize broader, better-sampled target responses with fixed architecture;
  defer representation experiments. See results/dflash-findings.md.

- 2026-10-11: Continued work moved to the consolidated `koogle/0-to-llm` repository
  after discovering the source repository's migration notice. Restored the existing
  budget ledger before launching jobs; preserved prior reservations and Modal resources.
- 2026-10-11: Completed matched fresh 768/6144-response DFlash fits at 20000 updates.
  More data improves held-out acceptance (0.2185 to 0.3844) and reduces overfitting,
  but larger-set training loss also remains high. Keep architecture fixed while
  resolving optimization/data coverage; no sufficient dataset size or broad speedup
  established. New test prompts remain sealed. See results/dflash-data-scaling.md.

- 2026-10-11: User prioritizes better block predictions, not speed, and requests
  a 2B Qwen comparison. Add exact Qwen3.5-2B as a quality-only target; disclose
  family/tokenizer/draft-capacity changes and unsupported hybrid cache rollback.
  See docs/dflash-qwen35-2b.md.

- 2026-10-11: Completed the requested Qwen3.5-2B quality experiment at 20000
  updates and 6144 responses. Selected shared-source-prompt matches are 0.170
  versus 0.387 for 0.5B; 2B training remains weak too. Do not interpret this as
  a pure size effect or solve it by speed work. Preserve the negative result,
  keep test sealed, and prioritize reliable drafter fitting/training fidelity.
  See results/dflash-qwen35-2b-findings.md.

- 2026-10-11: User requests sustained small-target improvement studies based on
  the DFlash comparison. Return to frozen Qwen2.5-0.5B, distinguish its size from
  the separately trained drafter, and prioritize quality. Predeclare QK-norm and
  depth comparisons, a held-out multi-token quality gate, and bounded adaptive
  follow-ups in docs/dflash-improvement-series.md. No new budget authorization.

- 2026-10-12: Correct ambiguous target-size labels: the historical 2B-target
  experiment trained a separate approximately 105M drafter; the current frozen
  Qwen2.5-0.5B improvement study trains a separate 79M drafter. Neither target
  is being trained or used as the draft network.

- 2026-10-12: Completed 6144/24576-response exposure comparison improves selected
  validation NEW from 0.574 to 1.022, below the gate. Both selected full-training
  audits show generalization gaps; the smaller final head memorizes nearly exactly.
  Compare 24576/49152 responses from identical selected 79M weights with matched
  optimization, retaining all old training groups and sealed held-out data.
  Larger final-checkpoint audit continues independently. See the improvement protocol.
