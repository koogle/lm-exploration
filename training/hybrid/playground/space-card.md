---
title: Hybrid 720M playground
emoji: 🧪
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Hybrid 720M playground

Text continuation from a from-scratch ~720M hybrid decoder (12 Kimi Delta Attention
and 4 full-attention layers, GLM tokenizer). The checkpoint and its training
provenance are described in `model-settings.json`; the code lives in
[`training/hybrid`](https://github.com/koogle/lm-exploration/tree/main/training/hybrid)
of lm-exploration. It runs on CPU with incremental decoding (a KV cache for the
full-attention layers and the recurrent state for KDA layers).
