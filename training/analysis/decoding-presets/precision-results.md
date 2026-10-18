# Full-FP32 sensitivity check

| Setting | Samples | BF16 repeat token4 | FP32 repeat token4 | Token-identical completions |
|---|---:|---:|---:|---:|
| greedy | 10 | 77.12% | 77.20% | 9 |
| topk50_t0.1 | 20 | 75.72% | 74.92% | 14 |
| topk50_t0.8 | 20 | 21.36% | 22.88% | 1 |
| baseline_t1_p95 | 10 | 6.32% | 4.32% | 0 |
| t085_p95_rp110 | 20 | 3.48% | 4.68% | 0 |

Same exact model, prompts, BOS prefix, seeds, and completion cap. FP32 disables TF32 for CUDA matmul and cuDNN. Matching BF16 outputs are retained from the original studies. Native token repetition is comparable within this model; this does not establish universal model quality.
