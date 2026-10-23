---
configs: []
---
# Qwen3.5 distillation training shards

Prepared from the private dataset `jfrick/qwen35-python-math-trajectories-20261022`
(revision recorded in `manifest.json`) by
[`distillation/validation`](https://github.com/koogle/lm-exploration/tree/main/distillation/validation)
in lm-exploration. Only accepted records that ended with the stop token are kept;
generation prompts are dropped.

- `shards/train-NNN/` (training order) and `shards/heldout-NNN/` (2% salted path-hash
  split, with `records.jsonl` problem/code/explanation text) each hold NumPy arrays:
  `tokens` (int32 native Qwen3.5 IDs), `offsets` (int64 document boundaries),
  `topk_ids` / `topk_logprobs` (teacher top-16 per position; int32 / float16, raw
  log probabilities, not renormalized), `counts` (native ID frequencies), plus
  `summary.json` with SHA256 hashes.
- `vocab-table.npy` / `vocab-kept.npy`: the 16,384-entry trimmed vocabulary
  (PAD 0, BOS 1, UNK 2, then the most frequent training IDs).
- `byte-lengths.npy`, `tokenizer.json`, `manifest.json` (paths are the original
  Modal volume paths; resolve them relative to this repository).

Semantic correctness of the teacher's solutions is not established.
