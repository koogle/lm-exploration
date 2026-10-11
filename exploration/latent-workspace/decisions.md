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
