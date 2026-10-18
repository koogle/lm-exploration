# Scholarly/code pretraining plan and agent handoff

Updated 2026-10-16. This is the current plan for the next from-scratch model.
It supersedes earlier 100M proposals and undecided mixture notes for this experiment.
The [machine-readable plan](scholarly-code-150m.json) records token budgets;
the [tokenizer recipe](../tokenizer/recipe-v1.json) pins all 12 source revisions
and shard lists used for the tokenizer sample.

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
[matched GPU profiling](../results/scholarly-124m-gpu-comparison-20261017.md)
covers one/two/three A10, L40S and H100 GPUs. Full corpus preparation, corpus audits
and substantial pretraining remain pending. The [distributed launcher](../distributed_training/README.md) now supports explicit
multi-GPU checkpoint/resume across Modal jobs. It remains separate from the
throughput-only benchmark; full-corpus restart costs and long-run operations are
not yet measured.
Instruction tuning and optional later reasoning training are separate experiments.

## Dataset mixture

Apply these proportions to **loss-bearing tokens under the frozen tokenizer**
in model training. FineWeb-Edu is capped at 10%; math and papers together are 35%.
The high-level mix is agreed; subdivisions within math/papers and books below are
working allocations to validate against usable capacity.

| Source | Share | At 20B | At 50B | Initial dataset |
| --- | ---: | ---: | ---: | --- |
| FineWeb-Edu | 10% | 2B | 5B | HuggingFaceFW/fineweb-edu |
| Open-source code | 30% | 6B | 15B | codeparrot/codeparrot-clean-train |
| FineMath | 20% | 4B | 10B | HuggingFaceTB/finemath, finemath-4plus |
| peS2o | 10% | 2B | 5B | common-pile/peS2o_filtered |
| PubMed/PMC | 3% | 0.6B | 1.5B | common-pile/pubmed_filtered |
| arXiv | 2% | 0.4B | 1B | common-pile/arxiv_papers_filtered |
| Books and textbooks | 25% | 5B | 12.5B | Sources below |

The tokenizer's book allocation is Pressbooks 3%, LibreTexts 3%, DOAB 4%,
Project Gutenberg 5%, pre-1929 books 7%, and Library of Congress 3% of the whole
sample. All use the corresponding `common-pile/*_filtered` repositories pinned
in the recipe. Keep books at 25% overall; adjust child shares explicitly after
measuring capacity rather than silently exhausting and repeating small sources.
Institutional Books 1.0 is a possible expansion, researched but not integrated.

The initial code source is Python-heavy. It is a bootstrap source, not evidence
of broad programming-language coverage or sufficient eligible capacity for 15B
exposures. Retain only records whose license metadata matches MIT, Apache-2.0,
BSD-2-Clause, BSD-3-Clause or ISC. Preserve repository, path, revision and license
provenance. Evaluate additional openly licensed code collections if capacity or
language diversity requires it; do not silently change this policy.

Use FineWeb-Edu, not Cosmopedia. Prefer FineMath and peS2o within the technical
allocation. Dataset filtering scores and scholarly provenance are quality signals,
not proof that every claim or solution is correct. FineMath-4plus is a subset of
3plus; do not count them as disjoint corpora.

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

## Tokenizer: code-focused 32K default

The repository default is [`structured-code-v1`](../tokenizer/structured-code-v1/),
a 32,768-entry byte-level BPE trained on approximately 64 MiB on Modal.
The user selected this tokenizer for the model's code and tool-calling focus.

- Fitting mixture: 40% Python, 20% JSON, 20% Git diffs, 20% tool-call formats by bytes.
- Same complete byte alphabet and pretokenization rules as the original tokenizer;
  no normalization, automatic boundaries, padding or truncation.
- Special IDs: PAD 0, BOS 1, EOS 2, UNK 3. Model vocabulary size must be 32,768.
- Exact artifact SHA256: `add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`.
- Published at [jfrick/scholarly-code-tokenizer](https://huggingface.co/jfrick/scholarly-code-tokenizer).
  The adjacent `publication.json` pins the immutable Hub revision.
- Compared with the earlier 2 GiB / 32K default: 13.97% fewer tokens on tool-format
  fixtures, 11.83% fewer on diffs, 8.21% fewer on JSON and 2.81% fewer on Python;
  13.99% more tokens on the scholarly mixture. This trade-off was accepted explicitly.
- JSON/diff/tool examples are controlled generated fixtures, not production traces.
  See the [full comparison](../results/tokenizer-structured/structured-code-v1/REPORT.md).

The old `scholarly-code-v1` 16K assets and all prior measured candidates remain
immutable historical versions. Existing encoded shards/checkpoints are not compatible
with the new token IDs. Use new staging/snapshot versions and retokenize before model
training; the loaders reject mismatched tokenizer identities. Do not change an old
checkpoint's vocabulary metadata to pretend it uses this tokenizer.

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
[jfrick/fineweb-edu-scholarly-gpt-10m](https://huggingface.co/jfrick/fineweb-edu-scholarly-gpt-10m),
revision `320d1fe54c611756317ad0a7bbeeb87c686325c3`.
It is a historical experiment, not the initialization for this from-scratch model.
No new 150M model training was launched. The next implementation milestone is
**a verified, resumable tokenized corpus plus a profiled approximately 150M model**.
