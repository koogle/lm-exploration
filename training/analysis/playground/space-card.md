---
title: ClimbMix 124M playground
emoji: 🧮
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# ClimbMix 124M playground

One prompt, several 124M models from lm-exploration, side by side: pretraining end points
of the ClimbMix + code warmup-stable-decay run and the models fine-tuned from them.
`models.json` records each checkpoint, its SHA256, what training it had and its benchmark
scores. The app and export code live in
[`training/analysis/playground`](https://github.com/koogle/lm-exploration/tree/main/training/analysis/playground).
