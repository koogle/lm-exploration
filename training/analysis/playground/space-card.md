---
title: ClimbMix 124M base vs math SFT
emoji: 🧮
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# ClimbMix 124M: base vs math SFT

One prompt, five 124M models from lm-exploration: the 5.45B-, 10.9B- and 20B-token
end points of the ClimbMix + code warmup-stable-decay run, and the 5.45B and 20B end
points after one epoch of answer-only SFT on GSM8K + MetaMathQA. `models.json` records
each checkpoint, its SHA256 and its benchmark scores. The app and export code live in
[`training/analysis/playground`](https://github.com/koogle/lm-exploration/tree/main/training/analysis/playground).
