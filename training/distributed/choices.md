# Configuration audit — PR #13

Scope: settings in the shared subword trainer, dedicated small-model workflow and
completed-card publication. This reconstructs decisions from the checked-out branch;
the original implementation session is unavailable. Verdicts describe the code
before the configuration fixes requested on 2026-10-21.

Resolution: all five unsound choices below are addressed locally. Training recipes
now select the loader and optimizer/diagnostic settings; the trainer reads data
constants from their owners; `workflows/small-5b.json` owns preparation and
continuation policy. Existing defaults are preserved, except incompatible loader
inputs now fail explicitly and model dimensions are no longer duplicated in the
workflow validator. Publication prose and evidence selection now live in Markdown;
verified reports supply measured settings and metrics. The local suite and targeted checks
pass; CUDA execution and deployment have not been requalified.

## Unsound

- **Publication facts and prose embedded in executable code (PR #13; confidence:
  high).** Rebuilding the card always prints one experiment's model size, training
  count, loss and descriptions, even if its input reports change. The script also
  assumes three training segments. The publication source was not separated from
  the renderer, so future edits could silently publish stale facts. Corrected
  decision: Markdown owns prose and artifact selection; verified measured reports
  supply recorded settings and metrics. Historical facts missing from those
  reports remain explicit documentation, with no fallback to today's defaults.
- **Manifest-selected loader (PR #13; confidence: high).** Launching the same
  training recipe against raw text silently changes the loader implementation;
  against token shards it selects another implementation. The recipe omitted the
  choice, leaving reproducibility dependent on an input-file branch. This affects
  every future snapshot format. Corrected decision: the recipe selects the loader;
  that loader validates its input and rejects incompatible formats.
- **Hidden optimizer and diagnostic settings (inherited trainer; confidence:
  high).** Changing AdamW's decay averages (betas), gradient clipping strength,
  number of validation windows or console cadence requires editing Python.
  These choices were not exposed alongside learning rate and evaluation cadence.
  Future comparisons would inherit them silently. Corrected decision: validated
  recipe fields own these values and checkpoints record the resolved settings.
- **Duplicated data-contract constants (trainer; confidence: high).** The model
  assumes a vocabulary of 32,768 and capacity calculation assumes two passes.
  Those values already belong to the frozen tokenizer and snapshot manifest.
  Independent literals can drift when their owners change. Corrected decision:
  read the tokenizer constant and the validated snapshot policy.
- **Workflow-specific settings in Python (PR #13; confidence: high).** Preparation
  selects a particular corpus recipe and concurrency, while the controller fixes
  qualification steps, session bounds and model dimensions. Changing an experiment
  can require editing executable logic as well as its training recipe. Corrected
  decision: workflow configuration owns preparation and orchestration settings;
  the training recipe owns model dimensions, with resource compatibility checked
  independently.

## Sound

- **Explicit review opt-in (PR #13; confidence: high).** An unreviewed snapshot
  fails unless the recipe explicitly permits it. A small model or pilot does not
  imply permission. This keeps future experiment recipes honest about pending
  audits; retain the flag and the original manifest evidence.
- **Frozen tokenizer and bounded source reuse (existing data contract;
  confidence: high).** Both loaders reject incompatible tokenizer identities and
  snapshots outside the supported two-pass policy. A recipe cannot safely change
  token meanings or claim additional unique data. Retain these checks; broader
  tokenizer/repetition support requires coordinated preparation and loader work.
- **Strict resume identity (existing trainer; confidence: high).** A new recipe
  or changed implementation cannot silently continue an old checkpoint as the
  same experiment. Keep config, source, data and runtime-version checks. Existing
  measured runs retain their original source and config for exact resume.
- **CUDA execution and deployment safeguards (existing trainer/workers;
  confidence: high).** NCCL communication, BF16 computation, atomic checkpoints,
  no automatic retries and reserves before hard job termination define the
  implemented execution contract. They are not interchangeable experiment
  settings without supporting code and verification. Retain these boundaries.
