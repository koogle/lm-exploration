# Simple LLM

Simple LLM is my personal project for exploring language model design, training and inference.

- [Inference](inference/): browser experiments and custom Metal kernels for Apple hardware.
- [Training](training/): models from scratch and a Modal training-run UI; [public Shakespeare 10M](https://huggingface.co/jfrick/shakespeare-vanilla-gpt-10m).
- [Latent workspace](exploration/latent-workspace/): speculative-decoding research, completed training and qualification results.

Latent workspace: [Hugging Face model](https://huggingface.co/jfrick/qwen2.5-0.5b-dflash-drafter-79m) · [code](exploration/latent-workspace/src/latent_workspace/) · [learnings](exploration/latent-workspace/docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md) · [research PDF](exploration/latent-workspace/results/drafter-learnings.pdf) · [qualification results](exploration/latent-workspace/results/qualification-findings.md).

The goal: run a model I trained myself on my own hardware, using kernels we wrote together.

Commits run ruff on staged Python files; enable once per checkout with `git config core.hooksPath .githooks`.
