---
title: Distillation student playground
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Distillation student playground

Interactive greedy or sampled generation from a from-scratch student trained on
Qwen3.5-9B Python/math exercises. The model, its settings and its training provenance
are described in `model-settings.json`; the code lives in
[`distillation/playground`](https://github.com/koogle/lm-exploration/tree/main/distillation/playground)
of lm-exploration, and the training and evaluation in `distillation/validation`.

The student completes the training format: a problem description, then the code of a
`solve` function, then an explanation. It runs on CPU with a key/value cache.
