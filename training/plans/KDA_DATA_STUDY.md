# Why the foundation mixture is 50/40/10

Decision recorded November 6, 2026. The selected foundation mixture is **50%
ClimbMix / 40% permissive GitHub code / 10% Nemotron CC-Math score-3 data**.
Shares describe intended post-filter token exposure, not source rows or bytes.
This is a coding-focused compromise supported by a small directional study,
not a demonstrated optimum or a reproduction of another model's private recipe.

| Source | Share | Conditional 100B-token allocation |
| --- | ---: | ---: |
| [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | 50% | 50B |
| [GitHub Code Clean](https://huggingface.co/datasets/codeparrot/github-code-clean), permissive licenses | 40% | 40B |
| [Nemotron CC-Math](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1), `3/` subset | 10% | 10B |

The math study used **score 3**, not the combined 3+ release (`3/` plus `4plus/`).
All arms used the same math files, so their comparison remains valid.
ClimbMix may itself contain code and math: source shares are not exclusive
semantic-category measurements.

## Evidence and the accepted tradeoff

Approximately 128M-parameter models trained for 200,015,872 tokens each, with
one seed, common held-out documents, tokenizer, 2K context, optimizer, MTP and
cooldown. The four-arm comparison varied mixer and source weights. The final
middle-ground arm held KDA fixed and changed only the weights.

| Model and mixture (general/code/math) | Code BPB | General BPB | Math BPB |
| --- | ---: | ---: | ---: |
| KDA 70/20/10 | 0.93620 | 1.15824 | 1.12703 |
| KDA 40/50/10 | 0.78313 | 1.23492 | 1.15558 |
| Full attention 70/20/10 | 0.91405 | 1.15326 | 1.12111 |
| Full attention 40/50/10 | 0.76837 | 1.22455 | 1.14831 |
| **KDA 50/40/10, selected** | **0.82666** | **1.20195** | **1.14534** |

Lower BPB is better. Raising code to 50% improved code prediction by about 16%
but worsened general-text BPB by 6–7%. The selected middle ground improves code
BPB **11.7%** relative to KDA 70/20/10, with **3.8% worse general** and **1.6%
worse math** BPB. It therefore **fails the proposed 2% general-retention guardrail**.
The user accepted this explicit tradeoff in favor of coding and selected 50/40/10.

Full attention was slightly better and faster at this size/context. Retaining
KDA is the chosen research direction, not a claim that the study established
its superiority. Single-seed, short small-model runs do not establish target-scale
convergence, executable coding ability or LiveCodeBench frontier performance.

[Archived study evidence](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/kda-data-study-20261105)
includes preparation, diagnostic attempts and final reports. Initial runs hit
preview-sampling compilation problems and were restarted from the original seeds
with previews disabled. The completed study uses the corrected runs. One control's
controller recorded a directory-exists error even though its trainer completed;
checkpoint and resumed numerical evaluations are retained with the raw status.

## Why these sources and what remains unqualified

ClimbMix preserves broad language coverage, real code supplies implementations,
and math supplies a controlled reasoning-relevant component. The three sources
were available in a common audited study corpus. A richer earlier proposal
included CC-Code and concentrated synthetic instruction; it was not the mixture
that produced these results. Accepted dataset terms alone do not qualify a new
source's quality or capacity, so Nemotron CC-Code has no reserved share in this
selected recipe. Nemotron Pretraining Code v3 requires code retrieval from its
metadata before it can be compared as a replacement source.

Code Concepts, STEM-SFT and reasoning math traces remain later-stage candidates.
Their inclusion or timing needs matched continuation comparisons and executable
verification where possible. No conclusion here establishes that synthetic data
is intrinsically inferior or that math automatically transfers to coding.

The sampled corpus used separated source shards, repository splits for code,
normalized exact-document deduplication and conservative benchmark screening.
Fork-aware near-deduplication, global cross-shard deduplication, generated/vendor
code filtering and representative language coverage remain limitations. Broad
corpus qualification must precede scale-up; study quality is not proof of final
corpus readiness. See [pretraining constraints](KDA_960M_PRETRAINING.md).
