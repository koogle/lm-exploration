# 124M - 10.879B: final validation and repetition

Training completed on October 19, 2026. The final checkpoint is step 66,065, with 4,999,971,081 additional targets and 10,878,562,619 lifetime targets. The model archive is private: https://huggingface.co/jfrick/124M-10.879B .

## Matched validation

Lower cross-entropy loss (nats/target) is better. These are the same 7,620,187 held-out targets from 2,782 source-family selections, scored in FP32 with TF32 off. Windows are at most 2K; this does not establish full-8K quality.

| Source | Original 5B | EU parent | Final 10.879B |
|---|---:|---:|---:|
| Python code | 1.559 | 3.994 | 1.421 |
| FineWeb-Edu | 2.675 | 2.911 | 2.700 |
| FineMath | 1.799 | 2.518 | 1.817 |
| peS2o | 2.313 | 2.565 | 2.330 |
| PubMed | 2.115 | 2.495 | 2.140 |
| arXiv | 1.950 | 2.809 | 1.963 |
| Pressbooks | 2.505 | 2.751 | 2.546 |
| LibreTexts | 2.244 | 2.555 | 2.293 |
| DOAB | 2.362 | 2.837 | 2.379 |
| Gutenberg | 2.819 | 3.437 | 2.848 |
| Pre-1929 books | 2.873 | 3.240 | 2.889 |
| Library of Congress | 2.881 | 3.269 | 2.896 |
| ECB | 2.548 | 2.065 | 2.494 |
| EMA | 2.475 | 1.766 | 2.379 |

Token-weighted loss on this fixed evaluation sample: 2.4019 → 2.8187 → 2.4030. This is a family-sampled evaluation mix, not the training mixture.

Coding improves over both previous checkpoints. The other 11 original sources recover from EU forgetting but remain slightly worse than the original 5B model. ECB and EMA lose most EU-specific gains, while retaining small gains over the original model.

## Repetition

150 matched continuations: ten fixed prompts, three checkpoints, one greedy seed and two seeds per sampled setting. Cap: 256 new tokens; stop on EOS. All generation uses FP32 with TF32 off; PAD/BOS/UNK are suppressed, EOS is allowed. No repetition penalty or n-gram blocking is applied. Prompts cover general text, maths, story, three Python functions, ECB and EMA. This is a small diagnostic, not a general capability benchmark.

Each cell is the mean fraction of generated token four-grams that repeat an earlier four-gram in the same continuation. Lower is less exact repetition.

| Decoder | Samples/model | Original 5B | EU parent | Final 10.879B |
|---|---:|---:|---:|---:|
| Greedy | 10 | 80.2% | 80.2% | 81.6% |
| T=0.2, k=10 | 20 | 78.2% | 74.9% | 77.0% |
| T=0.8, k=50 | 20 | 33.0% | 40.9% | 23.5% |

Long exact loops (at least 64 tokens covered by at least three consecutive copies of a token pattern of length 1–64):

| Decoder | Original 5B | EU parent | Final 10.879B |
|---|---:|---:|---:|
| Greedy | 8/10 | 8/10 | 8/10 |
| T=0.2, k=10 | 16/20 | 14/20 | 13/20 |
| T=0.8, k=50 | 2/20 | 6/20 | 0/20 |

Repetition is **not solved**: greedy and very low-temperature decoding remain strongly repetitive. T=0.8/k=50 is less repetitive in this small matched sample, and improves versus both parents. Zero flagged long loops does not mean no repetition: near-repeated paragraphs, renamed functions and semantically repeated sentences may evade that narrow flag. Ordinary code syntax also repeats naturally. Fewer repeated four-grams do not establish factual or programming correctness.

All raw samples remain private. The separate all-samples report includes every prompt, decoder, seed, continuation and repetition result; examples are not cherry-picked into the aggregate. Public repository artifacts contain aggregates only.

Validation uses paired family bootstrap intervals; interval and per-source counts are in validation-detailed-summary.json. They are conditional on selected families and do not account for selection of prompts or multiple source comparisons. Split reconciliation, near-duplicate and contamination audits remain pending.
