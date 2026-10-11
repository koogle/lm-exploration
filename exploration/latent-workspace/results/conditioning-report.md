# Trained conditioning comparison

![Layer probes](conditioning-layer-scan.png)

![Decoder training](conditioning-comparison.png)

Same 196k training windows and 192 validation windows; test withheld during selection.
Decoder comparisons share a final-state residual shortcut. Probe ranking
uses no such shortcut. All-masked loss is not sequence likelihood.

| Condition | Best update | NLL | Prose | Code | Translation | Qwen prefix / 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conditioning-parallel-all-01 | 4000 | 5.161 | 5.351 | 4.751 | 5.380 | 1.083 |
| conditioning-all-01 | 4000 | 5.259 | 5.437 | 4.888 | 5.450 | 1.099 |
| conditioning-best-01 | 4000 | 5.311 | 5.512 | 4.936 | 5.484 | 1.042 |
| conditioning-selected-01 | 4000 | 5.331 | 5.464 | 5.004 | 5.524 | 1.036 |
| conditioning-final-01 | 4000 | 5.390 | 5.605 | 4.989 | 5.576 | 0.979 |
| conditioning-attention-all-01 | 4000 | 5.419 | 5.596 | 5.120 | 5.541 | 1.104 |
| conditioning-attention-selected-01 | 4000 | 5.439 | 5.584 | 5.068 | 5.665 | 0.995 |
| conditioning-attention-best-01 | 4000 | 5.460 | 5.682 | 5.110 | 5.588 | 0.995 |
| conditioning-jspace-01 | 4000 | 5.532 | 5.632 | 5.195 | 5.769 | 1.031 |

Best checkpoints are selected by validation NLL within each run.
This exploratory ranking does not establish generalization to new source
groups, significance across many layer choices, or useful generation quality.
Raw JSON/JSON.gz preserves model/corpus/source revisions, per-example metrics
and checkpoint provenance. See ../docs/conditioning-study.md for the protocol.
