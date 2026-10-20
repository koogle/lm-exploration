# Repetition: Simple LLM 124M, 20.879B lifetime targets

Matched prompts, seed schedules and FP32; max 256 new tokens; no repetition penalty or ngram blocking. Repeated generated-token four-grams; long loop = >=64 tokens in >=3 tandem copies with periods <=64.

50 completions per checkpoint; a small diagnostic, not a broad capability benchmark.

| Decoding | Parent repeated 4-grams | New repeated 4-grams | Parent long loops | New long loops |
|---|---:|---:|---:|---:|
| Greedy | 81.6% | 82.8% | 8/10 | 8/10 |
| T=0.2, k=10 | 77.0% | 84.6% | 13/20 | 17/20 |
| T=0.8, k=50 | 23.5% | 25.5% | 0/20 | 2/20 |

New checkpoint: {"checkpoint": "/runs/simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt", "checkpoint_sha256": "ec6a6550f4f237d4c3c9530efe763061178440c09a8821265a67655c3bf8b85e", "phase_tokens": 9999963390, "step": 153342, "lifetime_tokens": 20878526009}

Parent metrics are from the previously saved matched FP32 generation suite; see summary.json for per-category results.

The additional 10B targets did not resolve repetition in this matched diagnostic.
Greedy decoding remains highly repetitive; low-temperature sampling produced more
long loops. Higher-temperature sampling is less repetitive, but remains imperfect.
The small difference at T=0.8 is not evidence of a statistically reliable change.
Improved held-out next-token loss alone did not predict reduced free-generation
repetition. This comparison does not isolate whether data, training objectives or
model capacity cause the behavior. Raw samples are retained in the private HF model
under `evaluation/repetition-20261020/` and the private Modal volume.
