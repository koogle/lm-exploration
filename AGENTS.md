# LM Exploration

This is a personal language-model learning and research scratchpad. Read the
relevant folder's README and instructions before making changes.

- `inference/web/`: historical Rust/WASM/WebGPU prototype and browser LM notes.
- `inference/phone/`: native Apple inference, handwritten Metal kernels, phone/iPad
  research and UI designs. Measure correctness and hardware performance separately.
- `training/`: from-scratch language models, dataset/architecture comparisons and
  a planned GitHub-connected training UI hosted on Modal. Begin with standard
  kernels and designs; custom training kernels are optional later work.
- `exploration/latent-workspace/`: existing frozen-Qwen and speculative-decoding
  research. Its AGENTS.md, uv environment, results and Modal resources are scoped
  to that folder, not the entire repository.

Keep the root README short. Put plans and run instructions in their owning folder.
Distinguish mockups, implemented behavior and measured results. Describe what code,
docs and pull requests do; do not state what was removed or is not used (that history
belongs in commit messages). Preserve experiment provenance and negative results. Run
project commands from the documented working directory. A repository consolidation
does not require GPU training or deployment.
Keep credentials and local checkpoints out of Git.

## No backward compatibility

This repository only moves forward. Code serves the current configs, architectures,
optimizers and data formats. Do not add or keep compatibility paths for old checkpoints,
result schemas, source hashes or launchers (legacy loaders, allowlists of old hashes,
format fallbacks, version switches). Old artifacts stay on HF with their provenance;
if one is needed later, write the compatibility code then, for that one use. Safety and
correctness checks against the current format stay.

## Modal job runtime

Every Modal job launched by an agent must stop automatically within 24 hours
(86,400 seconds) of launch, including detached and background jobs. Enforce this
through Modal's API/SDK before launch so shutdown survives agent disconnection.

- Set a server-side execution timeout of at most `24 * 60 * 60` seconds with
  `@app.function(timeout=86400, retries=0)` (or the equivalent class/function
  configuration). For sandboxes, use `modal.Sandbox.create(..., timeout=86400)`.
  Preserve shorter existing limits.
- A Function timeout covers execution time per attempt, excluding queue/startup
  time; retries reset it. For the overall 24-hour deadline from launch, including
  queued work, retries and child jobs, persist the deadline and call/sandbox IDs
  and use an independently running remote supervisor to cancel outstanding work
  through `modal.FunctionCall.from_id(call_id).cancel()` or
  `modal.Sandbox.from_id(sandbox_id).terminate()`. Track and cancel children
  explicitly; do not assume stopping a parent stops them.
- Bound each new call by the remaining budget and refuse work after the deadline.
  Keep retries disabled unless they respect that same deadline. Do not restart
  a timed-out job automatically to bypass the cap.
- `FunctionCall.get(timeout=...)` only limits how long the caller waits; it does
  not stop the remote job. Reminders, local timers and manual cleanup are not
  shutdown enforcement.
- Verify the effective Modal timeout and any remote deadline supervisor before
  launch. Save checkpoints and partial results early enough to survive shutdown.

See Modal's [timeout semantics](https://modal.com/docs/guide/timeouts),
[FunctionCall cancellation](https://modal.com/docs/reference/modal.FunctionCall#cancel)
and [Sandbox API](https://modal.com/docs/reference/modal.Sandbox).

## Source and artifact ownership

Git is the source of truth for executable logic, training/evaluation/preparation
and publication scripts, recipes/configuration, tests and small test fixtures.
Keep concise plans, run instructions and experiment summaries in Git. Hugging Face (HF) is the durable store for experiment outputs and design
artifacts: checkpoints, generated result JSONs, detailed metrics/audits, evaluation
outputs, samples, plots, reports, UI mockups and design exports. Reusable UI code
and design-generation logic belong in Git; their published previews/exports belong
on HF. Do not commit generated artifacts merely because they are small or JSON.

- Preserve provenance and negative results in the HF archive, with concise findings
  and links in the owning folder's documentation. Do not replace measured evidence
  with an unsupported summary or discard failed experiments.
- Do not keep per-file artifact indexes (paths, revisions, checksums) in Git for
  completed experiments; the HF repository and its commit history are the record.
  Docs name the HF repository and folder. Pin revisions only while an experiment
  is in progress and a script depends on exact inputs.
- Before removing an existing artifact from Git, verify that its exact bytes are
  available on HF and update every consumer and documentation link. Do not remove
  an unpublished or unverified artifact.
- Scripts take explicit artifact paths (downloaded into an ignored local directory).
  Do not require generated results to be checked into Git for evaluation,
  publication or report rebuilding.
- Keep active recipes/configs and small intentional test fixtures in Git; JSON is
  not itself an artifact classification. Never publish credentials, private data
  or unsafe checkpoint contents as part of an artifact migration.

This policy does not itself migrate existing files or authorize uploads, training
or deployment. Perform migrations explicitly, preserving existing experiment
identifiers and provenance.

## Reports

Building reports by concatenating strings in code (`lines += [...]`, f-string
tables, prose in `print`/`write_text`) is a red flag; avoid it. Scripts emit
structured results (JSON/CSV); render tables with a library or a checked-in
template. Write findings and narrative by hand in the owning folder's
docs, not in generator code. Prefer reusing or extending shared tooling over a new
per-study report builder.

See [migration notes](docs/consolidation.md) for imported histories and source PRs.
