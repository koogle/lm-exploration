---
title: Hybrid 720M playground
emoji: 🧪
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid 720M playground

Text continuation from a from-scratch ~720M hybrid decoder (12 Kimi Delta Attention
and 4 full-attention layers, GLM tokenizer). The checkpoint and its training
provenance are described in `model-settings.json`; the code lives in
[`training/hybrid`](https://github.com/koogle/lm-exploration/tree/main/training/hybrid)
of lm-exploration. It runs on a GPU: one forward pass reads the prompt, then each new
token is one replay of a CUDA graph with fused KDA and attention decode kernels.
