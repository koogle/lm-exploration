# Training from scratch

Learn how to train language models from scratch, from tiny baselines toward
roughly one billion parameters, and understand how dataset mixtures and model
architectures affect the results. Start with standard kernels and established
designs. Custom training kernels can come later.

## Current state

The [Run Board](designs/run-board-web.html) and [earlier training studies](designs/train-design-studies.html)
are interactive HTML mockups imported from
[web-inference PR #1](https://github.com/koogle/web-inference/pull/1).
All displayed runs and metrics are illustrative. There is no connected training
backend, GitHub integration or deployed Modal UI yet.

Open either HTML file locally, or run this from `training/`:

```sh
python3 -m http.server 5174 --bind 127.0.0.1
```

Then open http://127.0.0.1:5174/designs/run-board-web.html.

## Goal

Understand language-model training by training models from random initialization,
beginning small and exploring up to approximately 1B parameters. Compare dataset
mixtures and architecture choices with clear records of what changed and what
was learned. Use standard kernels and model designs initially. Writing custom
training kernels is a later option, not a prerequisite.

The end-to-end goal is to export one of these models and run it on our own Apple
hardware through the handwritten kernels in [phone inference](../inference/phone/).
The tokenizer, architecture, tensor shapes and weight format will need an explicit
contract between training and inference when that integration is implemented.

## Run Board on Modal

Use [the latest web Run Board](designs/run-board-web.html) as the visual starting
point. Host the actual UI on Modal, with GPU training jobs behind it and GitHub
as the source of model code.

The intended workflow is:

1. Select a GitHub branch or PR and resolve its exact commit before starting.
2. Configure model architecture and parameter count, tokenizer, dataset mixture,
   optimization settings, seed, token budget and bounded compute budget.
3. Start a run and inspect its state, loss curves, tokens processed, throughput,
   checkpoints and generated samples. Support stopping and resuming explicitly.
4. Choose a baseline run and compare other runs at equal token exposure. Show
   differences in model code, configuration and data alongside the measured results.

The design's comparison panel should make it easy to see whether a single change
can explain a result. When code, data and configuration all differ, present that
confounding clearly rather than implying a controlled experiment.

## Reproducible experiments

Each run should retain the resolved Git commit, code/config snapshots, dependency
versions, model and tokenizer configuration, pinned dataset revisions and mixture
weights, split definitions, seed, optimizer/schedule, tokens and updates processed,
hardware, checkpoint lineage, metrics and sample outputs. Compare on fixed
validation slices and reserve test evaluation for frozen choices.

Dataset and architecture controls are part of the run definition, not just labels
in the UI. Keep raw records and checkpoints available independently of the dashboard.

## Initial milestones

- A small standard-model baseline that trains, checkpoints, resumes and generates.
- A bounded Modal training job with persisted run metadata, metrics and artifacts.
- The Run Board connected to real runs, with explicit start/stop controls and
  GitHub commit selection.
- Controlled data-mixture and architecture comparisons, then larger parameter sizes.
- Export and correctness checks against the custom Apple inference implementation.

The initial architecture, tokenizer, datasets, exact model sizes, hardware and
compute budget remain to be chosen. Fold in the user's other-thread notes before
treating these choices as settled. The existing speculative-decoding project's
budget, frozen-backbone assumptions and Modal resource names do not define this
new from-scratch training effort.

Additional notes from earlier conversations can be incorporated here when provided.
The existing frozen-model drafting research lives in
[exploration/latent-workspace](../exploration/latent-workspace/).
