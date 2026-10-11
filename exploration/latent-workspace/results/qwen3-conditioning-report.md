# Trained conditioning comparison

Layer scan not part of this model comparison.

![Decoder training](qwen3-conditioning-comparison.png)

Same 196k training windows and 192 validation windows; test withheld during selection.
Decoder comparisons share a final-state residual shortcut. Probe ranking
uses no such shortcut. All-masked loss is not sequence likelihood.

| Condition | Best update | NLL | Prose | Code | Translation | Qwen prefix / 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-conditioning-all-01 | 4000 | 5.423 | 5.644 | 5.171 | 5.454 | 1.057 |
| qwen3-conditioning-final-01 | 4000 | 5.674 | 5.916 | 5.449 | 5.656 | 1.115 |

Best checkpoints are selected by validation NLL within each run.
This exploratory ranking does not establish generalization to new source
groups, significance across many layer choices, or useful generation quality.
Raw JSON/JSON.gz preserves model/corpus/source revisions, per-example metrics
and checkpoint provenance. See ../docs/conditioning-study.md for the protocol.
