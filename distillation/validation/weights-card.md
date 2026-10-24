---
configs: []
---
# Distillation validation checkpoints

Final weights from the distillation validation ladder in
[`distillation/validation`](https://github.com/koogle/lm-exploration/tree/main/distillation/validation)
of lm-exploration (campaigns `distill-validation-20261023-01`, `-02-covering` and
`-03-qwen-200m`). Results, grades and completions for every checkpoint are in the HF
dataset `jfrick/lm-exploration-results`, folder `distillation-validation/`.

- `<campaign>/students/<arm>/model.pt`: from-scratch students (PyTorch state dict with
  `config` and `settings`) for the modern decoder in `training/10m/model.py`. Trimmed
  and covering arms use a 16,384-entry subset of the Qwen3.5 tokenizer (rebuilt from the
  shards in `jfrick/qwen35-distillation-shards-20261023`); full arms use all 248,320 IDs.
- `<campaign>/control/<run>/model/`: Qwen3.5-0.8B-Base fine-tuned on the same stream
  (text-only `Qwen3_5ForCausalLM`, FP32 safetensors, servable by vLLM 0.32).

The teacher's solutions were not verified for semantic correctness. Base model license:
Qwen3.5-0.8B-Base (Apache-2.0).
