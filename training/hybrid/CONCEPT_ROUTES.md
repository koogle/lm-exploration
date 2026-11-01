# KDA memory-route follow-up

Frozen checkpoint: `kda-concepts-1793489611/checkpoint.pt`, the 98.3M model trained
from main `6d2f9ab` on 100,007,936 token positions. No new training is planned.
The goal is to resolve ambiguities in [the initial study](CONCEPT_PROBES.md),
not to confirm its interpretation by selecting new favorable heads.

## Questions

- Do the small effects survive FP32 inference and readout?
- Does information reach the answer through KDA recurrent state (K), the short
  convolution histories (C), or full-attention key/value caches (A)?
- Are these routes redundant or do combinations matter?
- Does the previously selected block-8/head-3/column-13 effect generalize from
  Paris/Rome to London/Berlin and Tokyo/Sydney with new entities and wording?
- Is it specific relative to colors, animals and boolean alternatives?
- Do the native read-channel examples remain coherent across distinct documents?

## Frozen evaluation protocol

Use main's native prefill and a dimension-independent FP32 probe decoder,
captured by main's CUDA-graph/cache driver, with FP32 weights/activations and
readout, without autocast and with Torch TF32 disabled. Compare selected cases to
a full forward pass. Also record BF16-trunk/BF16-readout, BF16-trunk/FP32-readout,
and FP32-forward baselines so changes in numerical precision are visible.
Self-patches must reproduce recipient log-odds; swapping all cache types must
reproduce donor log-odds to 1e-5. Decode/full-forward discrepancy must be below .01
nat for the explicitly qualified examples. Failure stops the study.

At the intervention boundary the factual prefix **and** 0/32/128 identical
filler tokens have already been processed. Then replay an identical query suffix.
This tests information retained after the filler, unlike the initial experiment's
intervention before the filler. Same-length counterfactual prefixes avoid position
confounds. All answer alternatives are single tokens, but observations are still
binary forced-choice preferences, not free-generation accuracy.

216 paired contexts: nine lexical contrasts, two templates, four new entities,
three distractor lengths. Test both intervention directions. Evaluate all eight
K/C/A subsets; each replaces that route at every relevant layer. Also patch each
of the three full-attention layer caches separately. Test the fixed city head and
column, an adjacent column with matched perturbation norm, and matched-norm random
directions within the same head/column. Seeds for the random directions are fixed; reverse transplants negate the same
random direction, so signed effects do not cancel by construction.
These interventions produce 6,912 rows. The head/column are never reselected on
these evaluation examples. Random directions are perturbation controls, not
on-manifold counterfactuals. Cache swaps themselves may be off-manifold when only
some routes are changed; analyze combinations and task competence accordingly.

Record raw log-odds, oriented transplant effects, the clean donor/recipient gap
and whether both alternatives were correctly preferred. Analyze by lexical pair,
template and entity; opposite directions and repeated lags are not independent
replications. Do not divide by tiny clean gaps. Findings remain specific to one
undertrained checkpoint until replicated across model seeds and training budgets.

The atlas examines fixed city-related and neighboring/control coordinates on up
to 100 distinct held-out documents per source (ClimbMix, code, math). Only one
positive/negative extremum per original document can appear in each coordinate's
top list. Prefix and following text are shown; following text did not cause the
activation. The decoder state column is not itself a neuron, and a raw `q^T S`
read also depends on the current query. Examples are hypothesis material, not
automatic concept labels or proof of Goodfire's broad cleanliness hypothesis.

## Execution and provenance

From `training/`, use `launch_kda_routes.py`. One H100, at most two hours from
launch (minus two minutes), worker timeout 6,900 seconds, retries zero, independent
CPU deadline supervisor and persisted worker IDs. Stop analysis before the deadline
and preserve partial results; no automatic relaunch of a timeout. Source hashes and
launch identity go in `artifacts/results/<run>/launch.json`. Structured outputs go
to Modal volume `lm-exploration-training/<run>`, then the private HF dataset
`jfrick/lm-exploration-results/training/results/<run>`.

The first route-probe launch `kda-routes-1793522731` stopped before producing
comparisons because main's production decode kernel requires 128-wide heads.
This 64-wide checkpoint uses a dimension-independent PyTorch probe recurrence;
its equivalence to full forwards and cache-transplant identities is qualified
before interpretation. The failed launch is preserved in the HF archive.

## Completed findings

Run `kda-routes-1793522897` completed all 6,912 intervention rows and the
300-document atlas. Results are archived in the private dataset
`jfrick/lm-exploration-results`, folder
`training/results/kda-routes-1793522897`. All self/full-cache identities passed;
the four decode/full-forward checks had maximum discrepancy 0.000563 nat.

The previously selected state column has a reproducible **Paris/Rome contrast**,
not demonstrated general city semantics. At zero filler its mean oriented effect
was 0.07494 nat, versus -0.00011 for London/Berlin and -0.00028 for Tokyo/Sydney.
The Paris/Rome effect remained 0.01498 after 128 filler tokens; the corresponding
matched-norm random-column mean was -0.00028 (mean absolute effect 0.01074).
Every Paris/Rome column transplant shifted toward the donor. These are repeated
directions/templates over four entities, not independent trial counts.

Full-attention caches carried most of the measured answer contrast. Exact
three-route Shapley allocations, averaged equally over the nine lexical pairs,
were:

| Filler tokens | KDA state | Convolution history | Full-attention caches |
| --- | ---: | ---: | ---: |
| 0 | 0.10811 | 0.02142 | 1.07769 |
| 32 | 0.05180 | 0.00033 | 1.59989 |
| 128 | 0.03174 | 0.00000 | 2.03187 |

Units are nats of oriented binary log-odds effect. These allocate a particular
cache-intervention contrast, not all information in the model. On the subset
where both counterfactual answers were correctly preferred, the 128-token
allocations were K=0.03601, C=0.00001, A=3.17940. Cat/dog was correctly preferred
in every tested paired context at all three lags; its KDA-only effects were
0.09323/0.04211/0.02731, versus attention-only effects 2.55377/4.04477/5.25573.
Thus attention dominance is not solely an artifact of failed city recall.
Boolean recall remained unsuccessful; city and other color/animal contrasts had
mixed competence, so preference shifts must not be reported as successful recall.

The atlas's selected column activated on varied natural passages, including
religion/politics, paleontology and genetics. Inspection did not establish a
coherent city concept. This is qualitative evidence from 300 documents, not a
formal rejection of semantic organization or Goodfire's broader hypothesis.

The next discriminating questions are entity binding (two entities, same city
words, change only one assignment), lexical versus semantic role (same city words
in non-location contexts), and whether attention refreshes KDA during filler.
The last requires interventions before filler or fixed-input propagation of a
state difference. Current after-filler swaps cannot establish autonomous KDA
retention, and near-zero convolution-cache effects do not show that the learned
convolution is dispensable during processing or training. One undertrained 98.3M
checkpoint and at most 128 distractor tokens do not establish behavior at 1–2B
parameters or long context. No new training was performed in this follow-up.
