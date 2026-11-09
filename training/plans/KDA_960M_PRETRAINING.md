# KDA foundation pretraining constraints

The selected model and stages are in [the training plan](CODING_REASONER_1B.md);
[the mixture study](KDA_DATA_STUDY.md) explains the 50/40/10 decision and its
accepted retention cost. This document defines the scale-up requirements.

## Corpus

Plan a conditional single-pass 100B-token schedule: 50B ClimbMix, 40B permissive
GitHub and 10B Nemotron CC-Math score-3 tokens. Freeze source revisions,
tokenizer identity, filtering and train/held-out partitions in the preparation
manifest. Require measured usable capacity of at least target tokens times
source weight; estimates size preparation but cannot authorize training. Do not
fill a short source with silent repetition or substitute a new source.

Require content-hash train/validation separation across sources and repository
separation for GitHub. Screen conservative 13-token overlaps against the frozen
benchmark manifests, including the six LiveCodeBench releases used in the study.
Inspect hits and rejection rates: this screen does not establish complete
contamination freedom. Preserve source-license restrictions, language coverage,
source exposure and repetition audits. Bound math preparation by Parquet row
groups and persist progress through the shared workflow infrastructure.

Global near-deduplication, fork-aware separation and generated/vendor-code quality
remain explicit gaps; the audit below measures their size.

## Corpus audit (November 9)

The complete `kda-foundation-100b-504010-v1` corpus is prepared on the
`lm-exploration-hybrid-data` volume. Its `capacity.json` measures usable
single-pass training tokens after license, split and screen filtering:

| Source | Train tokens | Needed | Margin | Screen drops (documents) |
| --- | ---: | ---: | ---: | ---: |
| ClimbMix | 61.26B | 50B | 22.5% | 132,919 (0.13%) |
| Permissive GitHub | 49.46B | 40B | 23.6% | 297,051 (1.0%) |
| CC-Math score 3 | 15.44B | 10B | 54.4% | 1,496,775 (10.6%) |

Capacity is sufficient for 100B tokens at 50/40/10 without repetition.

A sample of 14 upstream code files spread across the 450 the recipe reads
(1.78M rows, 1.15M permissive, 8.0 GB) characterizes the code source:

- **Ordering.** Upstream files are shuffled: each file has the same language mix
  (Python 6.2–7.3% of bytes) and about 59,000 repositories. Reading the first
  450 of 880 files is representative, and training order is not language-sorted.
- **Exact duplicates.** None within or across the 14 files, also ignoring
  whitespace. Per-shard exact deduplication is sufficient.
- **Near duplicates.** A coarse line signature (the eight smallest hashes of
  lines of at least 20 characters) matches 6.7% of bytes to an earlier document.
  The rate still rises with each added file (1.7% in the first, 10.0% in the
  fourteenth), so the full 449-shard rate is higher; this sample cannot bound it.
- **Language mix.** By bytes: HTML 15.8%, Java 15.1%, JavaScript 14.2%, C 9.3%,
  C++ 8.5%, **Python 6.8%**, C# 5.2%, PHP 5.1%, TypeScript 4.3%, Markdown 3.7%,
  Go 3.4%, CSS 3.0%. HTML, CSS and Markdown are 22.5%. At 40B code tokens this
  is about 2.7B Python tokens and 9B markup tokens, while the evaluation target
  (LiveCodeBench) is Python only. Byte shares approximate token shares.
- **Generated, vendored and data files.** Files over 100 KB are 0.8% of documents
  and 25.0% of bytes. Paths under vendor, dependency or build directories and
  bundled libraries are 8.9% of bytes. The largest repositories by bytes are
  `cdnjs/cdnjs` (1.8%), a generated benchmark site, parser-comparison HTML and
  spending data. Upstream's filters (mean line length, alphanumeric fraction,
  keyword search) leave these in.

The contamination screen's drops are mostly idioms, not benchmark text. In 20,000
code and 6,000 math documents the most frequent matching 13-token n-grams are
number runs (`1, 2, 3, 4, 5`, `0, 0, 0, 0, 0`), the quadratic formula and
`\mathbb{R} \to \mathbb{R}`; the sampled math drop rate is 11.2%. A 13-token
window covers few characters of digits or LaTeX. The screen stays conservative
for evaluation, and biases the math source against documents containing lists
and standard formulas.

Assessment for launch:

- Capacity, exact deduplication, split separation and shard ordering are
  qualified by the measurements above.
- Near-duplicate and fork separation affect how much distinct code the model
  sees and how optimistic held-out code loss is. They do not affect
  LiveCodeBench, and do not block a launch.
- The code composition is the material gap: about a third of code bytes are
  markup, oversized or vendored files, and Python is a fifteenth. A code source
  restricted by language, file size and path, drawing Python from all 880
  upstream files, is a recipe change that needs its own capacity measurement and
  a matched comparison against this recipe before it replaces it.
- A minimum character length for screen n-grams would restore most dropped math
  documents; math capacity does not depend on it.

## Filtered code sources (recipe v2)

`kda-foundation-100b-504010-v2` keeps the 50/40/10 shares and the prepared
ClimbMix and math shards, and prepares the 40% code share as two sources from
all 880 upstream files. `python` (4%) keeps Python only. `code` (36%) keeps every
language except Python, HTML, CSS and Markdown. Both drop files over 100 KB and
paths under vendor, dependency or build directories or named after bundled
libraries, then apply the license allowlist, the content and repository split
and the contamination screen. Measured capacity after preparation
(`kda-foundation-data-v2-20261109-01`):

| Source | Share | Train tokens | Needed | Margin | Held-out tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClimbMix | 50% | 61.26B | 50B | 22.5% | 5.34M |
| Python | 4% | 4.93B | 4B | 23.2% | 0.06M |
| Other code | 36% | 45.29B | 36B | 25.8% | 0.67M |
| CC-Math score 3 | 10% | 15.44B | 10B | 54.4% | 23.3M |

A 100B run reads 4.0B Python tokens, against about 2.7B under recipe v1, and no
HTML, CSS or Markdown. Of the other-code rows that pass the license allowlist,
the path filter drops 3.1M and the size filter 0.37M.

The Python held-out set is one shard's 1% content-and-repository partition:
55,643 tokens, about 50 documents. Held-out Python loss from it is noisy; use
it for trend only. The matched 128M mixture study measured the v1 code source,
so this recipe's effect on held-out code, general and math loss is unmeasured.
The 960M configuration reads recipe v2 and samples the MTP loss at fraction 0.25.

[Capacity reports, sample statistics, screen hits and the scripts that produced them](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/corpus-audit-20261109).

## Training and budget

Use exactly eight 80GB H100s, the shared trainer, frozen model/tokenizer and MTP
objective. Select execution details from matched real-data correctness and
performance checks. Full pretraining remains held during performance
qualification; hardware and MFU experiments belong to the stacked implementation
PR and do not alter this mixture decision.

Start with a bounded qualification segment and verify finite losses, per-source
held-out evaluation, checkpoint/resume and source consumption before extending
that checkpoint lineage. Preserve periodic checkpoints and numerical evaluation;
preview generation is optional and must not invalidate performance measurement.
No midtraining, SFT or RL stage starts automatically with foundation training.

Keep the main-pretraining allocation at $5,000 within the overall plan. Set the
final token commitment from actual dollars per token, including preparation,
CPU, memory, storage, evaluation and checkpoint overhead. Reconcile previous
spend before each continuation and preserve the later-stage reserve. A 100B-token
target is conditional on capacity, observed learning and remaining budget.

Every launch must have a server-side timeout, zero retries and an independent
remote deadline covering queued work and children, no later than 24 hours after
launch. Preserve shorter diagnostic limits. Publish measurements, negative
results and checkpoints in the existing HF/Modal artifact stores, with source
provenance; executable launch/configuration details belong to the implementation
review rather than this planning-only change.
