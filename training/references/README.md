# Training references

Source-backed reading notes and proposed experiments for our from-scratch models.
External results are attributed to their authors; proposals are not measured
results or changes to the agreed training plan.

- [KV cache, step by step](kv-cache.md): causal attention with a KV cache in the
  10M model, illustrated by prefill and decode passes over "hello".
- [A fix for softmax and cross-entropy — 2026-10-24](softmax-fix-2026-10-24.md):
  Softmax Collapse/Cancellation (Barron et al.), a tested PyTorch port in
  `trainer/stable_softmax.py`, and the CUDA kernel our trainer now uses for its loss.
- [Smallest useful model and pretraining data — 2026-10-23](pretraining-data-2026-10-23.md):
  size and token evidence for useful small models; recent web, code and synthetic
  datasets; data-constrained mixing; and a proposed
  matched data ablation after the failed 124M runs.
- [Mirendil, Recursive and Flapping Airplanes — 2026-10-21](lab-training-2026-10-21.md):
  agent-assisted research, data efficiency and hardware-aware model development.
- [Older megakernel results](megakernels-2025.md): ThunderMLA, batch-one Llama-1B
  and eight-H100 Llama-70B; benchmark scope, ablations and pinned code references.
