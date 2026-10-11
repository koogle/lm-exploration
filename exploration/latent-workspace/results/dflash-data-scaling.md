# More response data improves held-out drafting, but acceptance remains low

Completed on Modal L40S, 2026-10-11. Two fresh fits used the same initial weights,
architecture, seed, shared 192-response validation cache, and **20000 updates** each.
Only the balanced nested training subset changed: **768 versus 6144 responses**.
All training mixed prose, Python code and English-to-German examples. No test responses
or test metrics were generated.

**Result:** larger training data improves held-out prediction across all three domains.
The selected draft matches 0.384 new tokens consecutively, versus 0.218 for the smaller
fit. This is held-out improvement within these collections, but still much less
than a reliable multi-token accepted block. It is not a sufficient-data-size finding.

## Validation-selected checkpoints

| Training responses | Distinct response-token positions | Selected update | Validation NLL | Consecutive NEW matches / 7 |
| --- | ---: | ---: | ---: | ---: |
| 768 | 42,653 | 7,000 | 7.5426 | 0.2185 |
| 6,144 | 341,069 | 17,500 | 6.3597 | 0.3844 |

Selection maximizes accepted NEW tokens, breaking ties with lower NLL. The supplied
anchor is never counted as a draft success. Each fit consumed 480000 sampled anchor
blocks and 3360000 repeated new-token label exposures; this matches optimization
exposure, not epochs or wall-clock time. Training speed differed between workers.

![Matched data-size learning curves](dflash-data-scaling.png)

Dashed lines show the same 12 training diagnostic prompts; solid lines cover all 192
validation responses. The one-new-token reference line makes the absolute acceptance
level visible. It is not a promised speed break-even threshold.

## Fitting versus generalization at the final update

This table audits **every training response**, at up to eight fixed anchors/response,
not every possible anchor. It uses final checkpoints, not the selected ones above.

| Responses | Final train NLL | Final validation NLL | Final train new matches / 7 | Final validation new matches / 7 |
| --- | ---: | ---: | ---: | ---: |
| 768 | 0.1377 | 9.3788 | 6.8259 | 0.1474 |
| 6,144 | 5.9707 | 6.3464 | 0.4446 | 0.3799 |

The smaller model largely memorizes training responses while validation deteriorates.
The larger model has a much smaller train/validation gap, but its training predictions
are also weak. Therefore this study does not establish that 6144 responses are sufficient,
or isolate insufficient optimization from data/capacity limitations. Further fitting of
a diverse corpus remains an open factor; this is not a reason to resume J-space/layer sweeps.

## Paired validation evidence

- New-token acceptance, larger minus smaller: **+0.1660**, conditional 95% bootstrap interval **[+0.1343, +0.1993]**.
- NLL, larger minus smaller: **-1.1829**, conditional 95% bootstrap interval **[-1.2605, -1.1078]**.

Acceptance improves on 118 validation prompts, ties on 55 and worsens on 19. The 10000
paired bootstrap samples resample source groups within each domain and retain block
weights. These intervals are conditional on validation-selected checkpoints: they do
not account for model-selection bias, training-seed variation or unidentified document-
family correlation. They are not final test-set evidence.

| Domain | 768-response NLL | 6144-response NLL | 768-response new matches | 6144-response new matches |
| --- | ---: | ---: | ---: | ---: |
| prose | 6.641 | 5.679 | 0.330 | 0.529 |
| code | 7.484 | 6.537 | 0.230 | 0.375 |
| translation | 8.503 | 6.863 | 0.095 | 0.249 |

## Verified generation and timing

Twelve shared prompts (four/domain), 32-token cap, three warmed alternating-order
repeats, timing includes prefill. Both methods use FP32 arithmetic on the same stored
BF16 target weights. All twelve prompts matched target greedy output for both models;
the generated texts also agree across the two runs. The earlier BF16 parity limitation
remains unresolved. Training/teacher generation still use BF16.

| Responses | Matched prompts | Median prompt speed ratio (AR/speculative) | Sum of AR medians / sum of speculative medians |
| --- | ---: | ---: | ---: |
| 768 | 12/12 | 0.994x | 1.042x |
| 6,144 | 12/12 | 1.095x | 1.138x |

This suggests a modest improvement on this small latency check, not paper-scale speedup
or a production benchmark. One especially fast code continuation contains a long string
of zeroes. The example remains in the results; it must not be described as evidence of
broad coding performance. Workload, hardware execution variation, short outputs and
precision limit performance conclusions. Speculation reproduces Qwen; it does not
correct Qwen’s poor translations or repetitive outputs.

### Repetition diagnostic (post hoc)

Group validation responses by unique four-gram fraction below 0.8 versus at least 0.8.
This is a repetition proxy, not a quality label. Counts and block-weighted acceptance
are shown within each domain; this split was not used to select checkpoints.

| Domain | Four-gram fraction | Prompts | 768-response new matches | 6144-response new matches |
| --- | --- | ---: | ---: | ---: |
| prose | < 0.8 | 41 | 0.357 | 0.585 |
| prose | >= 0.8 | 23 | 0.283 | 0.429 |
| code | < 0.8 | 30 | 0.287 | 0.421 |
| code | >= 0.8 | 34 | 0.180 | 0.335 |
| translation | < 0.8 | 1 | 0.000 | 0.125 |
| translation | >= 0.8 | 63 | 0.096 | 0.251 |

## What DFlash demonstrated

[DFlash §5 and Appendix A](https://arxiv.org/html/2602.06036v1) describe around 800k
target-generated training responses, usually five draft/feature layers and sequences
up to 3072 tokens, with 512 sampled anchors per sequence. Its targets are substantially
larger than our 0.5B base model. It combines current target-feature conditioning,
inference-matched anchor masking, early-token-weighted loss and cheap parallel drafting.

The paper evaluates math, code and chat benchmarks; its controlled LLaMA experiment
also transfers from UltraChat/ShareGPT training to benchmarks including GSM8K and
HumanEval. This is evidence of cross-dataset utility rather than a lookup-only system.
It does not prove absence of all benchmark contamination or arbitrary-domain transfer.
The verifier preserves target behavior in the algorithm; draft generalization determines
acceptance and speed. Our compact two-layer Qwen2 adaptation is not a full reproduction.

## Data and remaining limits

The common pool has 6538 usable training responses and 192 validation responses after
dropping 35 short and 27 duplicate responses. The two nested subsets are balanced across
all three domains. The new 192 test prompts have no generated labels or evaluation.
Prior held-out groups remain quarantined. Code splits use repository groups, prose
document hashes and translation English-sentence hashes.

This study improves sampling over the older file-beginning pilot, but uses a bounded
streaming shuffle rather than globally uniform source sampling. It is still limited
by three source collections, short 64-token teacher responses, Python-only code,
sentence-level translation grouping and prompt-level near-duplicate screening. The
older 380-response pilot used different sampling/validation and is not the matched control.

Next priority: establish adequate optimization and data coverage for this same drafting
baseline, with held-out multi-token acceptance as the gate. Do not claim that more
data alone is guaranteed to solve it, or that the sufficient dataset size is now known.

## Reproduction and archived evidence

- [Predeclared protocol and reference comparison](../docs/dflash-data-scaling.md).
- [Prompt/split audit](dflash-data-prompt-audit.json) and [launch identity audit](dflash-scale-launch-audit.json).
- [768-response raw fit](dflash-scale-768-01.json) / [full-response audit](dflash-scale-768-audit-01.json).
- [6144-response raw fit](dflash-scale-6144-01.json) / [full-response audit](dflash-scale-6144-audit-01.json).
- [Paired analysis](dflash-data-scaling.json) and [checkpoint identity verification](dflash-scale-checkpoint-audit.json).
- Training source: `e3da59d`; data preparation: `46f7256`, both in the consolidated repository.
- 75 local tests pass; both actual GPU fits, all-response audits and FP32 rollout checks completed.
- Selected heads are downloaded and checksum-verified; full resumable checkpoints remain on Modal.
- Cumulative conservative reservations: **$226.50 / $500**, not a confirmed bill; all jobs in this study completed.

```bash
uv run --with matplotlib python scripts/plot_dflash_scaling.py \
  results/dflash-scale-768-01.json results/dflash-scale-6144-01.json
uv run modal volume get latent-workspace-qwen-runs \
  dflash-scale-6144-01/best.pt artifacts/dflash-scale-6144-best.pt
uv run latent-workspace draft --checkpoint artifacts/dflash-scale-6144-best.pt --prompt 'def fibonacci(n):' --tokens 32
```
