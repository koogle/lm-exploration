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
Distinguish mockups, implemented behavior and measured results. Preserve experiment
provenance and negative results. Run project commands from the documented working
directory. A repository consolidation does not require GPU training or deployment.
Keep credentials and local checkpoints out of Git.

## Modal job runtime

Every Modal job launched by an agent must stop automatically within 12 hours
(43,200 seconds) of launch, including detached and background jobs.

- Configure an explicit Modal timeout of at most `12 * 60 * 60` seconds on each
  function or sandbox before launching it; preserve any shorter existing limit.
- For jobs spanning multiple calls, retries or child jobs, enforce a shared
  12-hour deadline and automatically cancel outstanding work when it expires.
  Per-call timeouts alone must not let the overall job exceed the deadline.
- The shutdown mechanism must work without the agent, chat session or local
  terminal remaining connected. Do not rely on a reminder or manual cleanup.
- Verify the timeout and shutdown configuration before launch. Save checkpoints
  and partial results early enough to survive termination; do not automatically
  restart a timed-out job to bypass the limit.

## Source and artifact ownership

Git is the source of truth for executable logic, training/evaluation/preparation
and publication scripts, recipes/configuration, tests and small test fixtures.
Keep concise plans, run instructions, experiment summaries and artifact indexes
in Git. Hugging Face (HF) is the durable store for experiment outputs and design
artifacts: checkpoints, generated result JSONs, detailed metrics/audits, evaluation
outputs, samples, plots, reports, UI mockups and design exports. Reusable UI code
and design-generation logic belong in Git; their published previews/exports belong
on HF. Do not commit generated artifacts merely because they are small or JSON.

- Preserve provenance and negative results in the HF archive, with concise findings
  and links in the owning folder's documentation. Do not replace measured evidence
  with an unsupported summary or discard failed experiments.
- Reference artifacts using the HF repository ID/type, immutable revision, file
  paths and SHA256 checksums in a small checked-in index. Do not rely on mutable
  `main` links or assume that a model card contains the underlying result files.
- Before removing an existing artifact from Git, verify that its exact bytes are
  available on HF, record the pinned reference/checksum, and update every consumer
  and documentation link. Do not remove an unpublished or unverified artifact.
- Scripts must accept explicit artifact paths or use a documented download step
  that fetches and verifies pinned HF artifacts into an ignored local directory.
  Do not silently fetch the latest revision or require generated results to be
  checked into Git for evaluation, publication or report rebuilding.
- Keep active recipes/configs and small intentional test fixtures in Git; JSON is
  not itself an artifact classification. Never publish credentials, private data
  or unsafe checkpoint contents as part of an artifact migration.

This policy does not itself migrate existing files or authorize uploads, training
or deployment. Perform migrations explicitly, preserving existing experiment
identifiers and provenance.

See [migration notes](docs/consolidation.md) for imported histories and source PRs.
