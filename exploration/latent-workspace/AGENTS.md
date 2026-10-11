# Latent Workspace — Research Folder Context

These instructions apply to `exploration/latent-workspace/` within LM Exploration.
Run the commands below from this directory. Its frozen-backbone constraints apply
to these experiments; the separate `training/` folder covers from-scratch models.

## Product and scope

Read README.md, decisions.md, OPEN_WORK.md and docs/knowledge/INDEX.md at session
start. This is a self-contained research project within LM Exploration, unrelated to Age of Agents.
User direction, 2026-10-11: use an already trained small Qwen model, freeze its
weights, and test block generation conditioned on J-space, hidden states from
individual attention/transformer blocks, and selected combinations. Collaborative
generation and native backbone pretraining are outside the first experiment.

## Current priority (2026-10-11)

Build a credible **DFlash-style speculative drafter for frozen Qwen2.5-0.5B** before
further representation experiments. See [baseline protocol](docs/dflash-baseline.md).
Train on target-generated responses with random clean anchors and fully masked
future blocks. Measure accepted NEW tokens beyond the supplied anchor, verified
output correctness, and end-to-end latency. Defer J-space, workspace, hidden-state
prediction and layer-selection/dropout ablations. Standard fixed target-feature
conditioning is part of DFlash; it is not an auxiliary hidden-state generation task.

## Architecture and invariants

- Python/PyTorch; **uv** owns dependency resolution, commands and `uv.lock`.
- Qwen2.5-0.5B shares tokenizer, frozen token embeddings and output weights across
  experiments. Train only the block decoder and conditioning modules.
- Prefix extraction must never see target tokens. Split documents before windows;
  calibrate the J-lens on training prefixes only, never held-out data.
- Distinguish residual-stream post-block states from self-attention outputs.
- Preserve paper references, immutable model/data revisions, configuration,
  random seeds, measurements and uncertainty. Do not call a logit lens J-space.
- A smoke run does not validate a hypothesis or generation quality. Report
  actual target-token and context-token exposures separately.
- Modal uses dedicated app `latent-workspace-qwen` and volumes. Keep reservations
  fail-closed, jobs bounded, and credentials out of code, logs and Git.

## Workflow

Work directly on main; **push regularly** to the private consolidated repository.
Direct commits to main are authorized. Keep changes scoped, readable and portable.
Use `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and
`git diff --check` before a verified checkpoint. GPU changes require a bounded
Modal smoke run when access is available. Review for leakage, incorrect layer
indices, autograd/freeze errors, irreproducible resumes and misleading metrics.
Record blocked or unexecuted checks honestly. No game build/deployment applies.

## Documentation as project context

Use [.agents/skills/project-documentation/SKILL.md](.agents/skills/project-documentation/SKILL.md)
for implementation, debugging, planning, system interaction and documentation.
Follow **consult → work → learn/update → continue**. Consult relevant guides
before changes and reconcile them after durable findings, steering, milestones,
and before commits/pushes/handoff. Keep guides focused, indexed and source-linked.

README.md owns summary/setup/references; decisions.md concise choices;
OPEN_WORK.md current tasks/blockers; docs/experiment.md the experiment contract;
docs/operations.md run procedures; results/ measured evidence. Preserve unresolved
work. Label proposals, implemented behavior and executed verification distinctly.
Never store secrets or raw conversation dumps. Documentation updates within the
authorized work do not require separate approval.

## Expanded evaluation requirement (2026-10-11)

Maintain separate train/validation/test partitions for prose, code and translation.
Split code by repository, not file/window; deduplicate normalized content across
partitions. Use paired translation data with loss only on target-language tokens.
Use validation for model/layer choices; keep test metrics unread until choices
and trained checkpoints are frozen. Report domains and Qwen sizes separately.
The requested size sweep reaches approximately 1B parameters, retaining frozen
pretrained backbones. Record architecture/tokenizer differences across families.
