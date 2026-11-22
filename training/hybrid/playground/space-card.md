---
title: {name}
emoji: 🧪
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# {name}

From-scratch hybrid decoders (Kimi Delta Attention and full-attention layers, GLM
tokenizer) side by side: the Compare tab runs the same prompts on every export listed
in `model-settings.json`, as text continuation or a chat turn with or without
thinking. The code lives in
[`training/hybrid`](https://github.com/koogle/lm-exploration/tree/main/training/hybrid)
of lm-exploration. It runs on a GPU: one forward pass reads the prompt, then each new
token is one replay of a CUDA graph with fused KDA and attention decode kernels.
