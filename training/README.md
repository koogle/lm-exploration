# Training from scratch

A personal learning project: train small language models from random initialization,
understand how architecture and data affect them, then add instruction tuning,
reasoning and reinforcement learning. Use Modal for training and local Apple
hardware for development and eventual inference.

## Direction and scope

The user's goals are:

- Learn model architecture and its evolution through small, measurable experiments.
- Study useful ideas from Qwen, GLM, Kimi and other open research; do not lock the
  framework to one model family.
- Use general web text, appropriately licensed code and other licensed documents,
  with multilingual coverage.
- Start small and progress toward roughly 1B parameters using several thousand
  dollars' worth of available Modal credits. An exact spending allocation is not set.
- Eventually pursue GPT-4-level performance on a narrow, objectively testable task.
  The task and reference model/version remain to be chosen; broad parity is not a
  promised outcome.
- Run a model we trained ourselves on our own Apple hardware using the custom
  kernels in [phone inference](../inference/phone/).

Use standard kernels and established designs initially. Custom training kernels
are optional later work. The frozen-Qwen drafter in
[exploration/latent-workspace](../exploration/latent-workspace/) is a separate
research effort, with its own resources and budget.

## Shakespeare 10M publication

A separate [vanilla 10M definition and bounded Modal workflow](10m/README.md)
now has a [trained public model](https://huggingface.co/jfrick/shakespeare-vanilla-gpt-10m) with its complete definition and tokenizer.
The selected update is 1,800 (validation loss 1.54483 nats/character); the later
4,000-update head overfit and was not published.
Its L4 jobs, dedicated volume and bounded allocation are separate from latent-workspace.
See the owning guide and `results/shakespeare-10m-*.json` for executed results.

## Modal training recipes

Modal runs are launched from a JSON [recipe](recipes/README.md) naming the architecture,
registered dataset, training settings and GPU limits:
`modal run modal_train.py --recipe recipes/<name>.json --run-id <id> [--smoke]`.
The local UI below continues to use the 100M architecture.

## Current state

For a minimal NumPy example of forward passes, handwritten backpropagation and
gradient descent, start with [the XOR example](examples/README.md).

The first local implementation is available:

- [`100m/model.py`](100m/model.py): **architecture only**. It imports every module
  unchanged from [`10m/model.py`](10m/model.py), which keeps the explanatory
  comments, and only scales the defaults: 14 layers, width 768, 12 attention heads
  and context 128, approximately 100M parameters. The exact count depends on
  vocabulary/context. Run provenance hashes both files.
- [`cli.py`](cli.py): thin `lm-train` and `lm-generate` commands; parses settings and loads the trainer or a checkpoint.
- [`trainer/`](trainer/README.md): training configuration, core loop/evaluation,
  checkpoint loading/saving/resume, run provenance and shared sampling.
- [`data/loading.py`](data/loading.py): character tokenization, text loading,
  deterministic splits and batching.
- [`model_checks.py`](model_checks.py): reference and tiny-model validation.
- [`ui.py`](ui.py) and [`ui.html`](ui.html): a minimal localhost Run Board that
  starts one real local job, displays loss/progress/logs/samples and stops it.
- [`data/`](data/README.md): curated dataset definitions and fetching scripts.
  Tiny Shakespeare is a learning benchmark, not a selected research corpus.
- `runs/<id>/`: local artifacts, ignored by Git. Model configuration, metrics,
  checkpoints and dataset provenance remain available independently of the UI.
  Each save writes a new `checkpoints/step-NNNNNN.pt`; nothing is overwritten.

`training/` is a uv-managed project with `pyproject.toml`, `uv.lock` and a Python
3.12 pin. It has its own ignored `.venv`; the latent-workspace environment is
separate. uv installs the model folder as `model_100m` and the data folder as
`training_data`, so imports work without manual `PYTHONPATH` settings.

Run from `training/`:

```sh
uv sync --locked

# ~100M count checked on the meta device; actual training tests use tiny models.
uv run lm-checks

# Start the interface; open the URL printed in the terminal.
uv run lm-ui

# Fetch the local benchmark, then train with code-defined defaults; no flags required.
uv run python -m training_data.fetching.fetch_tinyshakespeare
uv run lm-train

# Sample from a trusted checkpoint (prompt characters must be in its vocabulary).
uv run lm-generate --checkpoint runs/<id>/checkpoints/step-000100.pt --prompt 'ROMEO:' \
  --max-new-tokens 300
```

`lm-generate` reads the model configuration and tokenizer from the checkpoint and
prints each character as it is sampled. It
runs on CPU by default (`--device mps` is optional) and seeds from the training
seed unless `--seed` is given. It samples from the full softmax; temperature and
top-k are not implemented.

Edit `TrainConfig` in `trainer/config.py` for the dataset, steps, batch size,
learning rate, seed, device and logging/checkpoint cadence. Model dimensions and
context length live only in `100m/model.py`, not in CLI flags or UI controls.
Defaults use the locally fetched `data/raw/tinyshakespeare.txt` on CPU, with each direct run
writing a fresh `runs/<uuid>/` directory. To use Apple acceleration, set
`TrainConfig.device = "mps"` on supported hardware. An interactive terminal shows
tqdm progress; the UI captures real JSON metric lines instead.

The UI starts at port 5174 and automatically tries subsequent ports if occupied.
Its Start button writes `runs/<id>/training.json` and invokes
`cli.py --config <path>` (the same entry point as `lm-train`). This is the optional
interface for run settings,
not an architecture selector. Stop requests checkpointing at an optimizer boundary.

FP32 ~100M training requires several GiB and CPU execution can be slow. Full-size
throughput and model quality have **not** been measured; setup/imports/checks do
not launch a full-size or paid run. Shakespeare is a learning corpus, not evidence
of broad language capability.

To resume, increase `TrainConfig.steps` (the total update target), retain the
original training settings, and run `uv run lm-train --resume <checkpoint>`. Resumed training continues in the
checkpoint's own `runs/<id>/` folder. Only load trusted checkpoints.
CPU continuation was tested within the same runtime; cross-device bitwise
reproducibility is not promised. The UI does not yet expose resume. The first
trainer reads text into RAM and is deliberately not a large-corpus system.

The historical [Run Board](designs/run-board-web.html) and
[earlier training studies](designs/train-design-studies.html), imported from
[web-inference PR #1](https://github.com/koogle/web-inference/pull/1), remain
interactive mockups with illustrative runs and metrics. Open those HTML files
locally to inspect the original designs. GitHub integration and Modal job controls in the UI, custom kernels and the larger
experiment framework are not implemented; recipe-based CLI training
runs on Modal.

## Staged research plan

The stages and numerical ranges below synthesize proposals from the supplied
conversation. They are starting hypotheses, not approved run budgets, measured
results or thresholds that guarantee model quality.

| Stage | Proposed scale | Work | Evidence before advancing |
| --- | --- | --- | --- |
| 0. Fundamentals | 1–10M parameters; a 10M pilot | Minimal decoder-only GPT: embeddings, learned positions, causal attention, MLP, LayerNorm and next-token prediction. Establish data loading, checkpointing, resume and sampling. | Correctness against a reference, tiny-set fitting, reproducible resume and measured Modal throughput. |
| 1. Architecture | Around 30–50M | Introduce RoPE, RMSNorm, SwiGLU, GQA and QK normalization one at a time. Begin with a standard dense baseline. | Controlled changes improve loss, stability, memory or time to a fixed quality target. |
| 2. Data and language foundation | 150–300M | Compare tokenizer choices, language balance, code/text mixtures, filtering and optional fill-in-the-middle training. | Held-out loss and task scores improve across the intended languages and domains, not just training loss. |
| 3. Instruction and reasoning SFT | Reuse the selected 150–300M base | Separate instruction-following fine-tuning from reasoning/worked-solution fine-tuning. | Improvements on fixed instruction, coding and mathematics evaluations relative to the same base. |
| 4. Verifiable-reward RL | Reuse that checkpoint lineage | Start with arithmetic; progress to executable code and formal constraints. Compare an appropriate GRPO- or PPO-style approach with supervised controls. | Better held-out solution accuracy at comparable compute, with reward hacking and response-length effects checked. |
| 5. Scale the winners | 600M to roughly 1B | Combine only supported improvements, then repeat pretraining and post-training for the chosen specialization. | Pilot throughput supports the budget, and smaller experiments justify the recipe. |
| 6. Own-model inference | Selected trained checkpoint | Export tokenizer, architecture and weights to the Apple inference project. | Reference-output agreement before device-specific kernel optimization. |

The transcript also suggested 1.7B and larger models. Those remain possible later
extensions, beyond the current roughly-1B consolidation scope.

For architecture comparisons, train fresh models under matched data and evaluation
conditions. For post-training comparisons, fork the same base checkpoint so gains
cannot be attributed to different pretraining. Recheck promising results across
seeds before spending heavily on a larger run.

## Architecture experiments

Keep the implementation small and auditable, with explicit configuration for
attention, feedforward layers, residual/normalization choices and training objective.
Modularity should support comparisons; it does not imply every combination is valid.

1. **Baseline:** causal dense attention, then a Qwen-inspired decoder with RoPE,
   RMSNorm, SwiGLU, GQA and QK normalization. Randomly initialize model weights.
2. **Optimization:** compare AdamW with Muon as a separate experiment.
3. **Information flow and attention:** investigate Attention Residuals and hybrid
   linear/full attention, including Gated DeltaNet/KDA-style candidates.
4. **Later experiments:** small MoE, sparse long-context attention, alternative
   residual schemes and auxiliary multi-token prediction.
5. **Post-training:** test how much improvement is possible from the identical
   pretrained base using SFT and verifiable rewards.

These are candidate research directions from the conversation. Verify the relevant
papers, official implementations and prerequisites before choosing a specific
variant. The transcript's release dates, version labels and claimed improvements
for “latest” Qwen/GLM/Kimi systems have not been independently verified here.

Use PyTorch and standard supported attention/optimizer kernels first. The transcript
proposed BF16, SDPA/FlashAttention where supported, activation checkpointing when
needed, and DDP/FSDP only when scale requires them. Initial 2K–4K context is a
proposal to benchmark, not a fixed requirement for the smallest toy model.

## Data and tokenizer

Build a reproducible corpus before committing to a large run. Candidate sources
from the transcript are:

| Source | Intended role | Selection work still needed |
| --- | --- | --- |
| FineWeb / FineWeb-Edu | General and educational text | Pin revisions, select quality filters and review underlying content eligibility. |
| FineWeb2 | Multilingual text | Choose target languages, balance the sample and check tokenizer efficiency. |
| CodeParrot GitHub Code | A manageable first code subset | Check current loading/access support and verify language/license metadata against source repositories. |
| The Stack v2 | A possible larger code corpus | Confirm access arrangements, upstream licenses, provenance and removal requirements. |
| OpenCoder / RefineCode | Code curation research and possible data | Inspect the processing recipe, access conditions and source provenance. |
| Wikipedia, Wikisource and licensed technical/math documents | Reference, multilingual and reasoning material | Review source-specific terms and preserve attribution. |

These are a shortlist, not an approved corpus or a claim that every record is
eligible. Public availability and a dataset-level license are not sufficient
evidence about each underlying document.

For each retained record, preserve source URL/repository, revision, license,
language, content hash, filtering decisions and training eligibility. Candidate
code licenses discussed were MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause and ISC;
the final allowlist and any obligations still need to be settled.

Remove generated, vendored, minified and duplicate code where appropriate; retain
useful documentation and test relationships. Apply exact and near-duplicate checks,
exclude benchmark solutions, and split code by repository before creating windows.
Check for credentials and personal information during corpus preparation.
Deduplicate across dataset categories as well as within each source.

Train or explicitly choose the tokenizer before the larger experiments. A balanced
multilingual/code BPE sample and 16K–32K vocabulary were proposed for small models;
compare 32K and 64K only if the larger model and languages justify it. Measure
bytes/characters per token by language and code, rather than judging only English.
Reusing an existing tokenizer is a valid explicit choice; model weights still start
from random initialization.

Mixture weights and target languages are not decided. The pasted transcript refers
to a mixture visualization but does not contain its numerical proportions, so no
ratios are inferred from it. Keep unique corpus size separate from total training
token exposures, including repeated epochs.

## Token and compute planning

There is no single token count that makes a model “great.” Use validation curves,
task performance and measured throughput to decide whether more data, more
optimization or a different recipe is justified.

Selected starting ranges proposed in the conversation:

| Model scale | Total pretraining-token exposures |
| --- | --- |
| 1–10M | Begin with 10–50M tokens. |
| 30–50M | 100M–1B; consider 1–3B only after the pilot. |
| 100–150M | 1–3B initially; 3–10B as a larger experiment. |
| 300–500M | 5–15B initially; 15–50B as an ambitious experiment. |
| 600M–1B | 10–30B initially; 30–100B as an ambitious experiment. |

The conversation's 300M/20B run was illustrative. Code-specific token ranges were
also suggestions for the code portion of a mixture, not extra tokens to add on
top of a total budget.

Before a substantial run, measure sustained tokens/second, peak memory,
checkpoint/evaluation overhead and cost on the actual Modal configuration.
The rough dense-model estimate `6 × parameters × training tokens` is a planning
approximation, not a runtime or price quote. Budget CPU, storage, preprocessing,
evaluation, failed runs and reserves as well as GPU time.

The transcript's $5,000 allocation and GPU-hour prices were assistant examples,
not a confirmed budget or current pricing. Choose explicit per-run and overall
bounds using measured throughput and current rates. Do not borrow the
latent-workspace research ledger or treat this plan as authorization to launch jobs.

## Post-training and evaluation

Keep three experiments distinct:

- **Instruction SFT:** conversation formatting and instruction following.
- **Reasoning SFT:** worked solutions and structured problem solving. The transcript
  suggested OpenR1-Math-220k as a candidate; inspect provenance and terms before use.
- **Verifiable-reward RL:** sampled rollouts scored by reliable outcomes, beginning
  with exact-answer arithmetic, then Python functions with hidden tests and
  algorithmic tasks with randomized cases.

The proposed SFT volumes were 10–100M instruction tokens and 10–200M reasoning
tokens. These are exploratory ranges, not requirements. RL is budgeted through
rollout generation and optimization compute, not a fixed static corpus size.
Start RL when the model already generates some successful solutions.

Compare base, instruction-SFT, reasoning-SFT and SFT-plus-RL checkpoints. Include
an additional-SFT control with comparable compute. Track pass@1/correctness,
reward, response length and generalization to unseen task families. Treat a
change in style or longer answers separately from a gain in accuracy.

Freeze evaluation definitions early:

- Held-out language-model loss by language and domain.
- Multilingual QA/translation/reasoning for the selected languages.
- Executable coding tests and exact-answer mathematics.
- Instruction formatting and constraint compliance.
- Tokens/second, peak memory, wall-clock convergence and quality per compute cost.

Use validation for choices and keep test evaluation for frozen decisions. For
the narrow parity goal, select a specific reference model/version, use matched
prompts and tools, report uncertainty, and audit contamination.

## Run Board on Modal

Use [the latest Run Board design](designs/run-board-web.html) as the UI starting
point. The intended application is a Modal-hosted web interface backed by bounded
training jobs and persistent run artifacts.

1. Select a GitHub branch or PR and resolve its immutable commit before starting.
2. Choose architecture, tokenizer, dataset mixture, seed, optimization settings,
   token budget and compute limit.
3. Start, observe, stop and explicitly resume runs; inspect curves, throughput,
   checkpoints and sample generations.
4. Choose a baseline and compare code/config/data differences beside measured
   results. Compare at equal token exposure and also report compute differences.

When several factors change, show that confounding rather than implying a
single-factor experiment. Retain each run's code/config snapshot, dependencies,
dataset revisions and splits, mixture weights, tokenizer, seed, hardware,
token/update counts, checkpoint lineage, metrics and outputs independently of
the UI.

## Next concrete milestone

The tiny local baseline and minimal Run Board are implemented. Bounded CPU checks
cover reference attention/gradients, causality, tiny-set fitting, reproducible
resume, real UI-driven training and cooperative stop/checkpoint continuation.
These checks demonstrate mechanics, not useful language capability.

Next: choose the first research data/tokenizer recipe and an explicit GPU/budget
bound, then implement and measure one bounded Modal throughput pilot before
scaling the experiment matrix. The separate 10M Shakespeare workflow has now launched bounded Modal jobs; larger research training remains unexecuted.

Still to decide:

- The narrow target task and reference-model evaluation protocol.
- Target languages, initial data sources, license policy and mixture weights.
- Tokenizer choice, exact first model configuration and context length.
- First Modal GPU configuration and explicit experiment/overall budget.
- Which architecture ablation follows the verified baseline.

## Source and status

Extracted on 2026-10-12 from the user-supplied transcript of
[“GPT 3.5 Size Explained”](https://chatgpt.com/share/6accc26c-fd60-83ea-9f52-34cc038c7857),
together with the repository-consolidation instructions. The transcript was read
from the supplied text, not from the inaccessible shared-page body. This README
separates stated goals from assistant-proposed recipes and unverified external
claims. No training runs, data ingestion or deployment were performed for that
original documentation update. The later local fundamentals/UI implementation
has been exercised with bounded CPU demonstration runs, not GPU training or
research-corpus ingestion.

## Linting

Ruff is a dev dependency configured in `pyproject.toml`. From `training/`, run
`uv run ruff check --fix .` and `uv run ruff format .`. The repository pre-commit
hook in `../.githooks/` checks staged Python files across the whole repository.
