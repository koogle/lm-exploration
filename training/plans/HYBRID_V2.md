# Hybrid v2: a sub-2B coding and agentic model

Goal: the best open model below 2B parameters for coding, including agentic coding
(issue → repository edit → tests pass), starting from the 720M hybrid
([HYBRID_720M.md](HYBRID_720M.md)) and growing only where measurements show size is
the limit.

## Where v1 stands (measured)

| | 720M v1 (40B tokens, SFT, RL) | Best reference ≤ 2B |
| --- | ---: | ---: |
| LiveCodeBench v6 pass@1 | 1.7% (SFT) | 6.9% (Qwen3-1.7B-Base); 6.3% at 0.6B |
| HumanEval / MBPP | 13.4 / 12.0 | 29.9 / 38.8 (Qwen3-0.6B-Base) |
| Context | 4,096 | 32K+ |
| Reasoning traces, tool use in repositories | none | — |

Qwen gains almost nothing from 0.6B to 2B on LiveCodeBench (6.3% → 6.9%) and jumps
at 4B (17%), so below 2B the gap is data and training, not parameters.

## Stages

Each stage has a gate; a failed gate stops the plan for a decision.

1. **Long context, 32K** (implemented on this branch; data preparation and run
   next). Continued pretraining of the 40B export on `hybrid-long32k-v1`: 2B tokens,
   60% long documents (repository-level permissive code, files of one repository
   joined in path order, ≥ 4,096 tokens; PG19 books) and 40% replay of
   `hybrid-40b-v1` read at 32K. Learning rate a fifth of pretraining, linear decay.
   *Gate:* held-out loss on long documents keeps falling with in-document position
   up to 32K (`loss_by_position`), and the short-context benchmarks stay within one
   point of the 40B export.
   *Measured:* 32K training fits one H100 at microbatch 1: 62,700 tokens/s, 31.9% MFU
   (synthetic 30K-token documents, 2026-10-28).
2. **Code-heavy base.** Continue pretraining with code at about 40% (permissive
   GitHub code plus the repository collections), math 15%, web 45%, for 40–60B more
   tokens at 32K. Before scaling, an ablation at 2B tokens compares 720M with a
   ~1.5B variant (24 layers, width 2048, same layer pattern).
   *Gate:* HumanEval/MBPP at or above Qwen3-0.6B-Base's 30/39, measured, before any
   larger run.
3. **Reasoning SFT** at 16–32K: thinking traces for code and math (smoltalk2's think
   splits, OpenCodeReasoning-style data, GLM-5.3-written traces kept only when their
   code passes tests), then the existing RL loop with thinking on.
   *Gate:* LiveCodeBench v6 above 6.9% (Qwen3-1.7B-Base).
4. **Agentic SFT.** Multi-turn trajectories on PR→task environments
   ([pilot](HYBRID_720M.md#pr-to-task-pilot-2026-10-28), 506 verified tasks):
   GLM-5.3 acts as the agent with read/search/edit/run-tests tools inside the
   verification sandboxes; only trajectories whose final patch makes the
   fail-to-pass tests pass are kept, rendered in GLM's tool-call format.
5. **Agentic RL.** The same sandboxes give the reward (fail-to-pass tests pass, no
   regressions), with a curriculum from single-file to multi-file tasks.

## Evaluation

- LiveCodeBench v6, HumanEval/MBPP (and EvalPlus) every stage.
- Held-out PR tasks from repositories never used for training (split by
  repository), resolved rate.
- SWE-bench Verified (easy subset) and Terminal-Bench easy tasks once stage 4 runs.
- Long context: `loss_by_position`, plus retrieval of a value planted early in a
  32K prompt.

## Data needs

- More repositories with tests (Python and Go first): the MIT collector runs again
  with batched uploads; a fresh GitHub token triples its search rate.
- Licenses: permissive code only for pretraining (github-code-clean allowlist, MIT
  collection); PG19 is public domain; GLM-5.3 outputs (MIT-style weights license)
  may be used to train.

## Mid-training run (`hybrid-mid32k-v1`)

One run addresses the three v1 gaps together: context (long documents at 32K), code
(a code-heavy share of unseen permissive GitHub files) and reasoning (thinking traces,
OpenThoughts3 math, code and science traces from smoltalk2, rendered in GLM's chat
format with every turn's `<think>` kept). Sources prepared by `hybrid-long32k-v1` are
reused through `prepared_in`. 3B tokens from the 40B export with 30% replay of
`hybrid-40b-v1`; config `hybrid/configs/hybrid-720m-mid32k.json`.

| Non-replay source | Share |
| --- | ---: |
| Repository-level code (≥ 4,096 tokens) | 35% |
| PG19 books | 10% |
| Unseen permissive GitHub files | 30% |
| Thinking traces (OpenThoughts3) | 25% |

### Result: `hybrid-720m-mid32k-01` (2026-10-28)

8×H100, 2,861 steps, 3.0B tokens in 1.7 hours (median 513K tokens/s, 31.4% MFU), from
the 40B export. Long-document loss falls with in-document position, so the model uses
context up to 32K; the replayed short-document sources improved too.

| Held-out source | Loss, step 200 → end | Loss by position, 0–1K → 16–32K (end) |
| --- | --- | --- |
| Repository code | 1.044 → 0.901 | 1.183 → 0.610 |
| Unseen GitHub files | 1.002 → 0.886 | 1.160 → 0.341 |
| PG19 books | 3.178 → 2.868 | 2.945 → 2.838 |
| Thinking traces | 1.949 → 1.647 | 1.281 → 1.819 (rises: later trace text is harder) |
| ClimbMix / code / math (replay) | 2.636 / 1.019 / 1.794 → 2.575 / 0.894 / 1.724 | — |

Benchmarks (same harness and shots as the 40B record), 40B export → mid-training:

| HellaSwag | ARC-E | ARC-C | PIQA | WinoG | OBQA | SciQ | LAMBADA | MMLU | GSM8K | HumanEval | MBPP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51.4 → 48.7 | 63.4 → 60.7 | 35.4 → 32.8 | 72.7 → 73.0 | 57.4 → 55.6 | 38.8 → 38.0 | 87.8 → 85.9 | 41.2 → 43.9 | 26.4 → 24.8 | 3.5 → 2.4 | 4.3 → 6.1 | 5.0 → 8.0 |

LiveCodeBench v6 (base prompt): 0/175, as for the 40B export.

**Gate not met.** Code improved (HumanEval +1.8, MBPP +3.0) and long context works,
but commonsense benchmarks fell 2–3 points (HellaSwag, ARC), beyond the one-point
limit. Web text was 21% of training tokens here (ClimbMix inside the 30% replay)
against 70% in the 40B run. Next, before more stages: raise replay to 50% and keep
the code share by lowering books, compared on the same 3B tokens.

A second, accidental launch of the same run trained independently for about 500
steps, then failed out of memory before its first checkpoint; it wrote no
checkpoint and the recorded metrics are the completed launch's only.

Outputs (metrics, configs, capacity, benchmarks, LiveCodeBench):
`training/results/hybrid-720m-mid32k-01/` in `jfrick/lm-exploration-results`.
