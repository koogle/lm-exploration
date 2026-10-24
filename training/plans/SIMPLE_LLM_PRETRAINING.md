# Simple LLM pretraining plan and agent handoff

Updated 2026-10-16; dataset mixture replaced 2026-10-23. This is the current plan
for the next from-scratch model. It supersedes earlier 100M proposals and undecided
mixture notes for this experiment.

## Goal and scope

Train a decoder-only language model from random initialization, targeting roughly
**150 million total parameters including embeddings**. Seek coherent short prose,
technical explanations and elementary code/math continuations, and measure how
quality improves with additional data. Coherence is an empirical goal, not a
promised capability at a particular size. No teacher distillation is planned.

Use **20 billion subword-token exposures** as the initial substantial-run target,
with **50 billion** as a conditional extension if measured gains justify compute.
These are exposures, not claims of unique available tokens. At 150M parameters,
these correspond to about 133 and 333 tokens per parameter. The intended model
is deliberately trained beyond a rough compute-optimal token allocation.
The earlier references to 100 billion parameters were not the selected scale.

As of 2026-10-17, the merged modern architecture has 124,389,632 parameters,
the [subword pilot](../100m/PILOT.md) has completed 4.08M exposures, and
[matched GPU profiling](https://huggingface.co/datasets/jfrick/lm-exploration-results/blob/main/training/results/simple-llm-124m-gpu-comparison-20261017.md)
covers one/two/three A10, L40S and H100 GPUs. Full corpus preparation, corpus audits
and substantial pretraining remain pending. The [distributed launcher](../distributed/README.md) now supports explicit
multi-GPU checkpoint/resume across Modal jobs. It remains separate from the
throughput-only benchmark; full-corpus restart costs and long-run operations are
not yet measured.
Instruction tuning and optional later reasoning training are separate experiments.

## Dataset mixture

The user withdrew the 12-source scholarly/code mixture on 2026-10-23 after the
124M runs in [what didn't work](../NEGATIVE_RESULTS.md) failed to give usable
generations. The [data review](../references/pretraining-data-2026-10-23.md)
found curated web mixtures and filtered code better than these sources.
The recipe is [`climbmix-code-v1`](../data/recipes/climbmix-code-v1.json):

| Source | Share | Pinned input | Replaces |
| --- | ---: | --- | --- |
| ClimbMix | 85% | `karpathy/climbmix-400b-shuffle`, the copy nanochat trains on; first 200 of 6,543 text-only shards | FineWeb-Edu, FineMath, peS2o, PubMed, arXiv and the Common Pile books |
| Curated open-source code | 13% | `jfrick/curated-oss-code-20261019` (private): 49 established projects, latest snapshot per project, 210 `raw/` shards | `codeparrot/codeparrot-clean-train` |
| MIT AI-attributed code | 2% | `jfrick/mit-claude-codex-code-20261019` (private): 36 repositories, 54 `raw/` shards | |

Neither source has document IDs, so all three use `"split_by": "row"`. Code shards
are interleaved across repositories so training never sits in one project.
Code is 15% because the goal is coherent text and math, not coding. The MIT collection holds only about 40M training tokens, so it gets 2%: its 2B-run share then needs 20M unique tokens under the two-pass limit. A first 85/10/5 split stopped in preparation because 5% needed 50M. ClimbMix has
little explicit math; math ability is to come from fine-tuning on GSM8K-style data
after pretraining.

ClimbMix's card says MIT, but NVIDIA's `nvidia/Nemotron-ClimbMix` is CC BY-NC 4.0.
The curated code includes GPL projects such as Linux. Both are acceptable for this
personal research project; published models must carry those restrictions.
The removed 12-source recipe is identical to the historical tokenizer's
[`simple-llm-v1/recipe.json`](../tokenizer/simple-llm-v1/recipe.json), which stays as
that tokenizer's provenance. Rewritten data is allowed but not planned for the
first run because generating it would take too long.

## First run and next steps (2026-10-23)

1. **Pretrain** the 124M model on 2B tokens with
   [`workflows/climbmix-124m-2b.json`](../workflows/climbmix-124m-2b.json):
   three H100s, about 8 hours of training, inside the 24-hour launch deadline.
2. **Look at what we get:** held-out loss per source, fixed-prompt samples and
   the repetition suite. Continue to 5B if the samples improve.
3. **Fine-tune for math** on GSM8K-style data, such as TinyGSM or teacher-generated
   traces, then score GSM8K, SVAMP and GSM-Symbolic by exact match. Screen
   training data against the test sets.

## Streaming, storage and capacity

Stream pinned remote shards incrementally on Modal; a full upstream download is
unnecessary. Cache bounded, immutable prepared shards on a persistent Modal Volume
and reuse them across worker restarts. Target existing data volume
`lm-exploration-mixture-data` and run/checkpoint volume `lm-exploration-training`,
using fresh versioned paths for this experiment. The new tokenizer corpus has
not yet been uploaded there by this handoff.

Before a substantial GPU run:

1. Measure accepted unique tokens per source with the frozen tokenizer, after
   license/quality filtering and deduplication. Published upstream token counts
   use other tokenizers and are not our usable-capacity measurement.
2. Split by parent book, paper or repository before chunking. Remove exact and
   near duplicates within and across sources, especially overlapping paper and
   book collections. Audit evaluation contamination before locking benchmarks.
3. Preserve source revisions, document IDs, parent IDs, hashes and filtering
   decisions. Persist stream cursors, shuffle RNG state and shard hashes so a
   restart does not masquerade as fresh data.
4. Record unique tokens separately from exposures and effective passes per source.
   Initial policy: at most two passes per source; expand eligible sources or
   explicitly revise the plan if capacity cannot support a target share.
5. Mix by actual trained token counts, mask padding, and document EOS/packing
   policy. Do not reuse the small pilot/tokenizer cache to reach billions.

## Tokenizer: mixture 32K default

The default is [`climbmix-code-v1`](../tokenizer/climbmix-code-v1/), fitted on the
new recipe's training split with one token per digit. It packs 25% more ClimbMix
text into each token than the earlier code-focused `structured-code-v1`, and is
slightly better on code; see the [tokenizer guide](../tokenizer/README.md#default-tokenizer).
Special IDs (PAD 0, BOS 1, EOS 2, UNK 3) and the 32,768 vocabulary are unchanged.
Snapshots and checkpoints made with earlier tokenizers must be retokenized.

## Training and evaluation sequence

1. **Restore context.** Read this plan and AGENTS.md; verify the frozen assets.
   Set explicit resource requests, concurrency limits and runtime bounds.
2. **Prepare corpus.** Implement or recover the streaming adapters and resumable
   packing pipeline, audit splits/provenance, then produce a measured capacity
   table and actual token-weighted sampling manifest.
3. **Define model.** Use standard kernels and a conventional dense decoder.
   Target about 150M total parameters; report the exact count, including whether
   embeddings/output weights are tied. Context 1,024 minimum, 2,048 preferred if
   profiling supports it. Final depth/width, position encoding, optimizer, learning
   rate, batch size and schedule are not yet frozen.
4. **Qualify execution.** Test loss masking, causal behavior, tiny-set fitting,
   checkpoint/resume, tokenizer identity, data cursor restoration and repetition
   counters. Profile the actual model/context/precision on Modal and measure
   sustained tokens/second, memory, checkpoint/evaluation overhead and cost.
   Set explicit time, token and spend bounds using the restored ledger.
5. **Run in stages.** Include an early smoke/pilot checkpoint before committing
   to the full target. Main evaluation milestones: 2B, 5B, 10B and 20B tokens;
   optional 35B and 50B after reviewing the 20B result. Preserve resumable state
   and immutable model/config/data provenance at each checkpoint.
6. **Measure returns.** Plot held-out loss and task quality against cumulative
   token exposures, wall time, GPU hours and actual cost. Report each domain and
   the weighted aggregate. Use matched held-out examples and fixed generation
   prompts/seeds/settings. Include prose coherence samples, executable elementary
   code tests and exact-answer math tasks, with contamination checks.
7. **Decide continuation.** Report improvements per additional billion tokens and
   per GPU hour, with uncertainty across examples and seeds where affordable.
   Compare checkpoints at consistent schedule phases or with matched cooldowns.
   Decide the numerical stopping threshold before the substantial run; it is
   not yet set. If gains flatten within measurement uncertainty or useful quality
   per cost becomes poor, stop. 50B is conditional, not an automatic commitment.

Use validation for recipe/checkpoint choices and keep final test sets untouched
until decisions are frozen. Broad capability and reliable reasoning are not
implied by lower loss alone. Generate actual samples from trained checkpoints.
Do not compare raw character-model loss to subword-model loss; use a common
byte-normalized measure on matched text if a cross-tokenizer comparison is needed.

## Handoff state and next agent's starting point

This PR intentionally contains the plan, pinned tokenizer recipe and frozen
artifacts. It does not commit the many prior experiment changes in the active
workspace. A fresh checkout can load the tokenizer immediately using the example
in [the tokenizer guide](../tokenizer/README.md).

The original workspace `/workspace/lm-exploration/training` also holds uncommitted
`build_tokenizer_corpus.py`, `train_tokenizer.py`, `tokenizer_checks.py`,
`data/subword.py`, `data/mixture.py` and related Modal mixture scripts. Review and
port those in a follow-up implementation PR if that workspace remains available;
a fresh checkout must not assume these files exist. Raw tokenizer text is ignored
under `runs/tokenizer-corpus-v1`; candidate tokenizers are under
`runs/tokenizer-candidates-v1`. Neither cache is included in Git or durably uploaded
by this handoff. The recipe and evaluation retain source pins and provenance,
but exact corpus reproduction also requires preserving the original collector.

The published earlier character model is
[published 10M Simple LLM model](https://huggingface.co/jfrick/fineweb-edu-scholarly-gpt-10m),
revision `320d1fe54c611756317ad0a7bbeeb87c686325c3`.
It is a historical experiment, not the initialization for this from-scratch model.
No new 150M model training was launched. The next implementation milestone is
**a verified, resumable tokenized corpus plus a profiled approximately 150M model**.
