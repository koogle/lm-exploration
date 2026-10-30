# N-gram hints on the hybrid model — 2026-10-29

A measured pilot of n-gram hints, the technique behind the 6.7-second NanoGPT
speedrun record. Conclusion: on our data the hints mostly retrieve text that recurs
verbatim in the training split. On the 76% of validation text with no such overlap
they gain 0.009–0.018 nats. We are not adopting them for the hybrid pretraining plan.

## The technique

The blog post [N-gram hinted language modeling](https://brunborg.com/n-gram-hinted-lm)
([KellerJordan/modded-nanogpt PR 367](https://github.com/KellerJordan/modded-nanogpt/pull/367),
[reproduction repository](https://github.com/hermabr/n-gram-hinted-lm)) gives each
position the tokens that followed its last 6, 10 or 20 tokens in an indexed corpus,
with their counts. The model receives the count-weighted embedding of those
candidates, scaled by a learned trust per cell (length × count bin × top-token-share
bin). It is added to the residual stream after the embedding, mid-stack and before
the output head. The record trains on 70.8M tokens with hints from already-seen
tokens. It evaluates with a 37B-token FineWeb hint corpus, moving validation loss
from 3.7454 (no hints) to 3.2739; with a 75M-token corpus the loss is 3.7219. The
post notes that 6.6% of validation tokens still have a 20-gram match after fuzzy
deduplication.

## Our pilot

- **Model:** the hybrid decoder's default configuration: 16 layers, width 1536,
  `[KDA, KDA, KDA, full]`, MTP, Muon/AdamW. With the 32K tokenizer it has about
  0.53B parameters (the plan's 720M assumes the GLM vocabulary).
- **Data:** `climbmix-code-v6` (snapshot `climbmix-124m-wsd20b-02`, pending audits,
  the same override as the 124M runs), 2K packed windows.
- **Arms:** baseline and hinted, identical seed, data order and schedule: 1900 steps
  of 64 × 2048 (248,697,079 targets), one H100 each, about 88 minutes and 47k
  tokens/s each. The hint lookup ran on the CPU and did not slow training.
- **Hints:** as in the record (lengths 6/10/20, 168 cells, warm trust scales at 10×
  learning rate, three injection sites). During training a step read only rows of
  earlier steps (up to 16 steps old). Lookups read a row's inputs up to the hinted
  position, never labels. Evaluation tables came from the training split only.
  Tables rebuilt offline from the training prefix reproduced the online index's
  validation hints exactly.
- **Evaluation:** the first 512 validation rows (1,047,109 targets), mean
  cross-entropy in nats.

## Results

| Model | Hints at evaluation | Hinted targets | Validation loss |
| --- | --- | --- | --- |
| Baseline | none | — | 3.1401 |
| Hinted | none | — | 3.1470 |
| Hinted | its training rows (249M tokens) | 10.9% | 3.0614 |
| Hinted | first 500M training-split tokens | 12.9% | 3.0473 |
| Hinted | first 1B | 15.4% | 3.0269 |
| Hinted | first 2B | 18.4% | 3.0038 |

As in the record, each doubling of the hint corpus lowers the loss further
(0.014–0.023 per doubling). Without hints the hinted model is slightly worse than
the baseline.

The gain is concentrated on text that recurs in the training split. We scored every
validation target and grouped targets by segment (one document's span within a
row). A segment's overlap is the share of its targets whose 20-token context occurs
verbatim in the 2B-token table:

| Segment overlap | Targets | Baseline | Hinted, own rows | Hinted, 2B | Share of the 2B gain |
| --- | --- | --- | --- | --- | --- |
| none | 76.0% | 3.2355 | 3.2262 | 3.2174 | 10% |
| up to 10% | 16.6% | 2.8099 | 2.6915 | 2.5970 | 26% |
| 10–50% | 5.8% | 2.9640 | 2.3743 | 2.0637 | 38% |
| over 50% | 1.7% | 2.6837 | 1.6180 | 0.5843 | 26% |

Per position, the 2B table matches the full 20-token context for 2.8% of targets.
Their loss falls from 2.87 to 0.31. Targets with no match at any length move from
3.230 to 3.227.

## Interpretation

- **Mostly retrieval:** the hints act as a lookup, not a better language model.
  About two thirds of the gain comes from the 7.5% of targets in segments that
  substantially recur in training text. Some of that recurrence is duplicated
  documents across the split; some is shared boilerplate (licences, common code).
  The 20-token test does not separate them.
- **Small on new text:** on clean text the gain is 0.009–0.018 nats, and the hinted
  model is no better than the baseline without hints.
- **When it would pay:** hints suit settings where the evaluated or served text
  recurs in an indexable corpus, such as code completion against a repository or
  serving with a large retrieval index. Our goal for the hybrid plan is general
  model quality, so it does not fit.

Not measured: tuning of the trust scales (they stayed near their starting values),
the 720M configuration with the GLM tokenizer, downstream benchmarks, and hint
corpora beyond 2B tokens.

## Artifacts

- Metrics, logs, both checkpoints (safetensors and configuration), hint tables and
  per-target overlap losses are in the private HF dataset
  [`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-109/hint-pilot-01)
  under `pr-109/hint-pilot-01/`.
- The implementation, Modal launcher, publication script and tests are in closed
  pull request [#109](https://github.com/koogle/lm-exploration/pull/109) (head `f82719c`).
