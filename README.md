# Language Modeling Exploration

A personal playground for learning about language modeling—from pretraining and post-training to inference kernels and speculative decoding.

## Featured release

- [Malama v0.0.1](https://huggingface.co/jfrick/malama-v0.0.1): 0.93B KDA/full-attention hybrid Python reasoner trained from scratch on ~112B total tokens ([model card](training/publication/cards/malama-v0.0.1/README.md), [playground](https://huggingface.co/spaces/jfrick/malama-versions), [app](training/hybrid/playground/)).

## Repository overview

- [`training/`](training/): From-scratch language model training on Modal (10M/124M GPTs and the 0.93B KDA hybrid Malama line), data preparation, SFT, RL, evaluation, and publication tools.
- [`inference/`](inference/): Native Apple inference with handwritten Metal kernels ([`phone/`](inference/phone/)) and an earlier Rust/WASM/WebGPU browser prototype ([`web/`](inference/web/)).
- [`exploration/`](exploration/): Speculative-decoding and frozen-model drafter experiments ([`latent-workspace/`](exploration/latent-workspace/)) and teacher rollout studies ([`teacher-rollouts/`](exploration/teacher-rollouts/)).
- [`distillation/`](distillation/): Dataset preparation, archiving, and validation pipelines for distillation experiments.
- [`docs/`](docs/): Research notes and repository history.

## Reference library

Source-backed reading notes, walkthroughs, and research surveys ([`training/references/`](training/references/)):

- [Kimi Delta Attention, step by step](training/references/kda.md) & [KV cache, step by step](training/references/kv-cache.md)
- [Thinking Machines Inkling 2 Mini](training/references/thinking-machines-inkling2-mini-2026-11-21.md) & [Inkling 2 Small](training/references/thinking-machines-inkling2-2026-11-07.md)
- [Marin 535B-A23B architecture](training/references/marin-535b-architecture-2026-11-11.md)
- [SFT task vectors after RL](training/references/sft-task-vectors-2026-11-05.md) & [Ricursive Ripple](training/references/ricursive-ripple-2026-11-05.md)
- [Reasoning training below 1B](training/references/reasoning-training-2026-11-04.md) & [Looped LMs (Dory / ELR)](training/references/looped-lms-2026-11-03.md)
- [Open-model size frontier](training/references/open-model-frontier-2026-11-01.md)
- [N-gram hints on the hybrid model](training/references/n-gram-hints-2026-10-29.md)
- [Mirendil Fern 0.1 and 10M-token context](training/references/mirendil-fern-2026-10-26.md)
- [StableReinforce (Reinforce + Score Centering)](training/references/stable-reinforce-2026-10-25.md)
- [A fix for softmax and cross-entropy](training/references/softmax-fix-2026-10-24.md)
- [Smallest useful model and pretraining data](training/references/pretraining-data-2026-10-23.md)
- [Mirendil, Recursive and Flapping Airplanes](training/references/lab-training-2026-10-21.md) & [Megakernels (2025)](training/references/megakernels-2025.md)
- [Post-training inference speedups (2026)](docs/research/post-training-speedups-2026/)
