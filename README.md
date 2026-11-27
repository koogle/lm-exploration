# LM Exploration

A personal playground for learning about language modeling—from pretraining and post-training to inference kernels and speculative decoding.

## Featured release

- [Malama v0.0.1](https://huggingface.co/jfrick/malama-v0.0.1): 0.93B KDA/full-attention hybrid Python reasoner trained from scratch on ~112B total tokens ([model card](training/publication/cards/malama-v0.0.1/README.md), [playground](https://huggingface.co/spaces/jfrick/malama-versions), [app](training/hybrid/playground/)).

## Repository overview

- [`training/`](training/): From-scratch language model training on Modal (10M/124M GPTs and the 0.93B KDA hybrid Malama line), data preparation, SFT, RL, evaluation, and publication tools.
- [`inference/`](inference/): Native Apple inference with handwritten Metal kernels ([`phone/`](inference/phone/)) and an earlier Rust/WASM/WebGPU browser prototype ([`web/`](inference/web/)).
- [`exploration/`](exploration/): Speculative-decoding and frozen-model drafter experiments ([`latent-workspace/`](exploration/latent-workspace/)) and teacher rollout studies ([`teacher-rollouts/`](exploration/teacher-rollouts/)).
- [`distillation/`](distillation/): Dataset preparation, archiving, and validation pipelines for distillation experiments.
- [`docs/`](docs/): Research notes and repository history.
