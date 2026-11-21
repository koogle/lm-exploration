# KDA foundation pretraining constraints

The selected model and stages are in [the training plan](CODING_REASONER_1B.md);
[the mixture study](KDA_DATA_STUDY.md) explains the 50/40/10 decision and its
accepted retention cost. This document defines the scale-up requirements.

## Corpus

Plan a conditional 100B-token schedule: 50B ClimbMix, 40B code and 10B Nemotron
CC-Math score-3 tokens. The 40B code tokens weight curated projects and starred
repositories above the rest of GitHub ([recipe v3](#quality-weighted-code-sources-recipe-v3)).
Freeze source revisions, tokenizer identity, filtering and train/held-out
partitions in the preparation manifest. Require measured usable capacity of at
least target tokens times source weight, counting each source once per pass its
recipe allows; estimates size preparation but cannot authorize training. A source
repeats only up to its recipe's `max_passes`, and the run stops when a source is
used up. Do not substitute a new source.

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

## Quality-weighted code sources (recipe v3)

Decision of November 10: within the unchanged 50/40/10 shares, read more Python
and more high-quality code, and prefer repeating high-quality code to reading
low-quality code once. `kda-foundation-100b-504010-v3` splits the 40% code share
into six sources. All use the recipe v2 language, size and path filters.

| Source | Share | Train tokens | Passes at 100B | Passes allowed |
| --- | ---: | ---: | ---: | ---: |
| Curated projects | 4.0% | 0.854B | 4.68 | 5 |
| MIT collection | 0.2% | 0.053B | 3.80 | 5 |
| Starred Python | 4.0% | 0.984B | 4.07 | 5 |
| Other Python | 3.5% | 3.946B | 0.89 | 1 |
| Starred other code | 20.0% | 7.434B | 2.69 | 3 |
| Other code | 8.3% | 37.855B | 0.22 | 1 |

- **Curated projects** are `jfrick/curated-oss-code-20261019`: the latest snapshot of
  53 established projects (Linux, LLVM, Elasticsearch, ClickHouse, VS Code,
  Kubernetes, PyTorch, Rust, CPython, Django and others). The source keeps each
  project's own license, which includes GPL projects such as Linux and Git.
- **MIT collection** is `jfrick/mit-claude-codex-code-20261019`: the latest snapshot
  of 50 recently active MIT repositories.
- **Starred** sources are GitHub Code Clean rows whose repository has at least ten
  stars in `ibragim-bad/github-repos-metadata-40M` (`watchers_count`; 1,246,491
  repositories qualify). **Other** sources are the remaining rows. A repository
  absent from that table counts as unstarred.

A 100B run reads 28.2B tokens from curated and starred sources, against about
6.7B of such code under recipe v2, and about 8B Python tokens against 4.0B.
Curated projects, the MIT collection and starred Python are read four to five
times and starred other code 2.7 times; the passes repeat the same shard order.
The smallest margins are starred other code (2.69 of 3 passes) and curated
projects (4.68 of 5).

Train/held-out separation is by content hash for the two project collections
and by content hash and repository for GitHub sources, as before. Each source's
held-out shard is small (curated 0.84M tokens of one Linux file's partition,
starred Python 9K tokens), so per-source held-out loss during training is a
trend only. `kda-code-quality-eval-v1` is the common evaluation set: one upstream
GitHub file that no recipe trains on, restricted to repositories in the held-out
partition and excluding the 103 curated and MIT repositories. It holds 88 starred
Python documents (0.13M tokens), 385 other Python (0.50M), 899 starred other
code (0.95M) and 4,564 other code (5.39M), and evaluates the v1 ClimbMix and math
held-out shards alongside.

### Matched 128M comparison of recipe v2 and v3

Two 127.6M-parameter KDA models trained for 200,015,872 tokens each on one H100,
identical except for the recipe (context 2,048, one seed, MTP loss fraction
0.25; workflows `study-128m-mix-v2` and `study-128m-mix-v3`). Both are scored on
the common evaluation set, in bits per byte:

| Evaluation set | Documents | Recipe v2 | Recipe v3 | Change |
| --- | ---: | ---: | ---: | ---: |
| Starred Python | 88 | 1.0125 | 0.9564 | -5.5% |
| Other Python | 385 | 0.9513 | 0.9081 | -4.5% |
| Starred other code | 899 | 0.8146 | 0.8153 | +0.1% |
| Other code | 4,564 | 0.8282 | 0.8494 | +2.6% |
| ClimbMix | | 1.1936 | 1.1916 | -0.2% |
| Math | | 1.1276 | 1.1201 | -0.7% |

Recipe v3 predicts held-out Python 4.5–5.5% better and leaves general text and
math unchanged. It predicts unstarred non-Python code 2.6% worse, and starred
non-Python code the same.

What this does and does not show:

- The Python gain follows the Python share: v3 trains on about twice as many
  Python tokens (about 8% of the stream against 4%). The test does not separate
  that from any effect of repository quality.
- It gives no evidence that starred code is better training data for held-out
  prediction. Starred non-Python code rose from about 6% to 20% of the stream,
  with curated projects on top, and held-out starred code did not improve; the
  loss on unstarred code follows its share falling from about 30% to 8%.
- Bits per byte measures prediction of held-out code, including code one may not
  want the model to imitate. It is not executable correctness.
- A 200M-token run reads the start of each source once. It says nothing about
  reading curated projects and starred Python four to five times in a 100B run.
- One seed per arm; the size of seed noise at this scale is unmeasured, and the
  starred-Python set has 88 documents.

The 960M configuration reads recipe v3: the comparison supports it for the Python
target and shows no cost outside unstarred non-Python code. The weighting toward
starred repositories is a stated preference, not a measured gain.

[Run reports, common evaluation and capacity reports](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/code-mix-study-20261110).

## Architecture options from the Marin notes (128M study)

Two options from the Marin 535B notes were compared against the recipe v3 control,
each a 127.6M-parameter KDA model trained for 200,015,872 tokens on one H100 with
one seed (workflows `study-128m-marin-control`, `-conv` and `-embnorm`):

- **Short convolutions** (`short_conv_size: 4`): causal depthwise convolutions on
  full-attention keys, values and output and on every MLP output, identity at
  initialization and reset at document boundaries; 51,200 extra parameters.
- **Embedding RMSNorm** (`embedding_norm: true`): an RMSNorm on the embedding lookup.

Held-out bits per byte on the common evaluation set:

| Evaluation set | Control | Short convolutions | Embedding RMSNorm |
| --- | ---: | ---: | ---: |
| Starred Python | 0.9594 | 0.9441 (-1.6%) | 0.9268 (-3.4%) |
| Other Python | 0.9093 | 0.8986 (-1.2%) | 0.8810 (-3.1%) |
| Starred other code | 0.8174 | 0.8058 (-1.4%) | 0.7841 (-4.1%) |
| Other code | 0.8512 | 0.8399 (-1.3%) | 0.8170 (-4.0%) |
| ClimbMix | 1.1925 | 1.1857 (-0.6%) | 1.1787 (-1.2%) |
| Math | 1.1212 | 1.1116 (-0.9%) | 1.1072 (-1.2%) |

The control differs from the earlier run of the same configuration
(`mix-study-v3-20261110-01`) by 0.07–0.32% per set, an estimate of run-to-run
noise from one pair. Both options are outside it on every set.

The training-loss lead over the control behaves differently. Over the five
tenths of the run sampled, the short-convolution lead is roughly constant
(0.015, 0.032, 0.033, 0.038, 0.029 nats). The embedding-RMSNorm lead shrinks
steadily (0.232, 0.133, 0.101, 0.087, 0.071 nats). A lead that shrinks like this
fits faster early optimization rather than a better model at the end; 200M tokens
cannot tell which.

Decisions:

- The 960M foundation run continues without either option; its source hashes and
  checkpoint lineage are fixed.
- Short convolutions go to midtraining as a matched comparison: continue the
  foundation checkpoint with and without them, with the convolutions added at
  identity so the starting model is unchanged. Loading a foundation checkpoint
  into a model with new parameters needs an explicit initialization path that
  doesn't exist yet.
- Embedding RMSNorm is not adopted: the measured gain may not survive a longer
  run, and it can't be added to a trained model without changing its function.
  It needs a longer matched run before a future pretraining run uses it.

[Run reports, metrics and common evaluation](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/marin-ablation-20261111).

## Training and budget

Use exactly eight 80GB H100s, the shared trainer, frozen model/tokenizer and MTP
objective with its loss sampled at fraction 0.25 (10.6–10.8% more throughput in the
[matched benchmark](KDA_MIXER_TIMING_STUDY.md#mtp-loss-fraction-november-9); its effect
on quality is unmeasured). Select execution details from matched real-data
correctness and performance checks. Full pretraining remains held during performance
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
