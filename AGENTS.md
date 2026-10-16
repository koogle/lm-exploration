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
Keep credentials, local checkpoints and budget ledgers out of Git. A restored
budget ledger is not required for user-authorized jobs. Set explicit resource,
concurrency and runtime bounds before launching them.

See [migration notes](docs/consolidation.md) for imported histories and source PRs.
