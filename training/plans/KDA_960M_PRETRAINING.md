# KDA 960M pretraining

The model and training stages are in [the training plan](CODING_REASONER_1B.md);
[the mixture study](KDA_DATA_STUDY.md) covers the 50/40/10 split and its retention
trade-off. This note documents the corpus decisions, 128M ablations, run sizing,
and loss forecast for the 960M foundation run.

## Corpus

Target: 100B tokens — 50B ClimbMix, 40B code, and 10B Nemotron CC-Math score-3.
The code share evolved across three recipes, ending on [recipe v3](#quality-weighted-code-sources-recipe-v3),
which weights curated projects and starred repos above general GitHub.

Data rules:

- Source revisions, tokenizer, filters, and train/held-out splits are frozen in
  the preparation manifest.
- Capacity must be measured, not estimated, before training. Sources repeat only
  up to their recipe's `max_passes` and the run stops if one runs dry (no silent
  substitution).
- Train/held-out separation uses content hashes across all sources plus
  repository separation for GitHub.
- A 13-token overlap screen runs against frozen benchmark manifests, including
  the six LiveCodeBench releases.

Known gaps — no global near-deduplication, no fork-aware split, and upstream
generated/vendored code — are measured in the audit below.

## Corpus audit (November 9)

The `kda-foundation-100b-504010-v1` corpus on `lm-exploration-hybrid-data`
(`capacity.json`) has enough single-pass tokens for 100B at 50/40/10 after
license, split, and contamination filtering:

| Source | Train tokens | Needed | Margin | Screen drops (documents) |
| --- | ---: | ---: | ---: | ---: |
| ClimbMix | 61.26B | 50B | 22.5% | 132,919 (0.13%) |
| Permissive GitHub | 49.46B | 40B | 23.6% | 297,051 (1.0%) |
| CC-Math score 3 | 15.44B | 10B | 54.4% | 1,496,775 (10.6%) |

A 14-file sample across the 450 upstream code files read by v1 (1.78M rows,
1.15M permissive, 8.0 GB) shows what is actually in the GitHub source:

- **Ordering:** Shuffled. Every file has roughly the same language mix (Python
  6.2–7.3% of bytes) and ~59,000 repos, so reading 450 of 880 files is
  representative.
- **Exact duplicates:** None within or across the 14 files, even ignoring
  whitespace.
- **Near duplicates:** A coarse line signature (8 smallest hashes of lines >= 20
  chars) matches 6.7% of bytes to an earlier doc, climbing from 1.7% in file 1
  to 10.0% in file 14. The full 449-shard rate is higher.
- **Language mix (by bytes):** HTML 15.8%, Java 15.1%, JavaScript 14.2%, C 9.3%,
  C++ 8.5%, **Python 6.8%**, C# 5.2%, PHP 5.1%, TypeScript 4.3%, Markdown 3.7%,
  Go 3.4%, CSS 3.0%. HTML, CSS, and Markdown make up 22.5%. At 40B code tokens,
  that is ~2.7B Python and ~9B markup, while LiveCodeBench is Python-only.
- **Oversized and vendored files:** Files over 100 KB are 0.8% of docs but 25.0%
  of bytes. Vendor, dependency, build, and bundled-library paths are 8.9% of
  bytes. Top repos by bytes include `cdnjs/cdnjs` (1.8%), a generated benchmark
  site, parser-comparison HTML, and spending data.

**Concerns:**

- Near-duplicates and forks inflate held-out code metrics and cut distinct code
  volume, but they don't affect LiveCodeBench.
- The 13-token contamination screen mostly catches idioms, not benchmark text.
  In 20,000 code and 6,000 math docs, the top hits are number runs
  (`1, 2, 3, 4, 5`, `0, 0, 0, 0, 0`), the quadratic formula, and
  `\mathbb{R} \to \mathbb{R}` (11.2% sampled math drop rate). That biases math
  against lists and standard formulas. A minimum character length on n-grams
  would fix most of it, though math capacity doesn't depend on it.
- The real problem is code composition: roughly a third of code bytes are
  markup, oversized, or vendored files, and only ~1/15 is Python. That drove
  recipes v2 and v3.

## Filtered code sources (recipe v2)

`kda-foundation-100b-504010-v2` (`kda-foundation-data-v2-20261109-01`) keeps
50/40/10 and the v1 ClimbMix and math shards, and rebuilds the 40% code share
from all 880 upstream files as `python` (4%) and `code` (36%, excluding Python,
HTML, CSS, and Markdown). Both drop files over 100 KB and
vendor/dependency/build/bundled paths (3.1M path drops and 0.37M size drops in
other-code after the license filter).

| Source | Share | Train tokens | Needed | Margin | Held-out tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClimbMix | 50% | 61.26B | 50B | 22.5% | 5.34M |
| Python | 4% | 4.93B | 4B | 23.2% | 0.06M |
| Other code | 36% | 45.29B | 36B | 25.8% | 0.67M |
| CC-Math score 3 | 10% | 15.44B | 10B | 54.4% | 23.3M |

At 100B tokens this reads 4.0B Python (up from ~2.7B in v1) and zero markup.

**Concerns:** The Python held-out shard is only 55,643 tokens (~50 docs), so
held-out Python loss is noisy and only useful for trend. The earlier 128M
mixture study ran on v1, so v2's loss effect was not separately measured before
moving to v3.

[Capacity reports, sample stats, screen hits, and scripts](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/corpus-audit-20261109).

## Quality-weighted code sources (recipe v3)

Decision on November 10: keep 50/40/10, push more Python and higher-quality
code into the 40% share, and accept repeating high-quality repos instead of
reading low-quality code once. `kda-foundation-100b-504010-v3` splits the code
share into six sources, all using v2's language, size, and path filters:

| Source | Share | Train tokens | Passes at 100B | Passes allowed |
| --- | ---: | ---: | ---: | ---: |
| Curated projects | 4.0% | 0.854B | 4.68 | 5 |
| MIT collection | 0.2% | 0.053B | 3.80 | 5 |
| Starred Python | 4.0% | 0.984B | 4.07 | 5 |
| Other Python | 3.5% | 3.946B | 0.89 | 1 |
| Starred other code | 20.0% | 7.434B | 2.69 | 3 |
| Other code | 8.3% | 37.855B | 0.22 | 1 |

- **Curated projects** (`jfrick/curated-oss-code-20261019`): 53 established
  projects (Linux, LLVM, Elasticsearch, ClickHouse, VS Code, Kubernetes,
  PyTorch, Rust, CPython, Django, etc.), retaining original licenses including
  GPL.
- **MIT collection** (`jfrick/mit-claude-codex-code-20261019`): 50 recently
  active MIT repos.
- **Starred vs. other**: GitHub Code Clean repos with at least 10 stars
  (`watchers_count` in `ibragim-bad/github-repos-metadata-40M`, 1,246,491
  repos). Unmapped repos count as unstarred.

At 100B tokens this reads 28.2B curated/starred tokens (vs. ~6.7B in v2) and
~8B Python (vs. 4.0B). Curated projects, the MIT collection, and starred Python
repeat 4–5x in the same shard order; tightest margins are starred other code
(2.69 of 3 passes) and curated projects (4.68 of 5).

Per-source held-out shards are tiny (curated is 0.84M tokens from one Linux file
partition; starred Python is 9K tokens) — trend only. For real comparisons,
`kda-code-quality-eval-v1` holds out one upstream GitHub file excluded from
training and excluding the 103 curated/MIT repos: 88 starred Python docs
(0.13M tokens), 385 other Python (0.50M), 899 starred other code (0.95M), and
4,564 other code (5.39M), scored alongside the v1 ClimbMix and math held-out
shards.

### Matched 128M comparison (v2 vs. v3)

Two 127.6M-parameter KDA models trained for 200,015,872 tokens each on 1x H100
(context 2,048, one seed, MTP loss fraction 0.25; `study-128m-mix-v2` and
`study-128m-mix-v3`), scored on `kda-code-quality-eval-v1` in bits per byte:

| Evaluation set | Documents | Recipe v2 | Recipe v3 | Change |
| --- | ---: | ---: | ---: | ---: |
| Starred Python | 88 | 1.0125 | 0.9564 | -5.5% |
| Other Python | 385 | 0.9513 | 0.9081 | -4.5% |
| Starred other code | 899 | 0.8146 | 0.8153 | +0.1% |
| Other code | 4,564 | 0.8282 | 0.8494 | +2.6% |
| ClimbMix | | 1.1936 | 1.1916 | -0.2% |
| Math | | 1.1276 | 1.1201 | -0.7% |

v3 improves held-out Python by 4.5–5.5%, leaves ClimbMix and math flat, and
hurts unstarred non-Python code by 2.6%.

**Concerns:**

- The Python gain tracks doubling the Python share (~8% vs. 4%) and does not
  isolate repo quality.
- There is no evidence here that starred code improves held-out prediction:
  starred non-Python jumped from ~6% to 20% with zero BPB gain, while unstarred
  loss tracks its drop from ~30% to 8%.
- BPB measures next-token prediction on held-out code, not executable
  correctness.
- 200M tokens only reads each source once; it doesn't test 4–5x repetition at
  100B.
- Single seed per arm, and starred Python is only 88 documents.

**Decision:** Ship v3 for 960M. It helps Python with no cost outside unstarred
non-Python code; the starred-repo weighting is a stated preference, not a
measured win.

[Run reports, common eval, and capacity reports](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/code-mix-study-20261110).

## Architecture options from the Marin notes (128M study)

Two options from the Marin 535B notes were tested against the v3 control on the
same 127.6M / 200,015,872-token / 1x H100 / single-seed setup
(`study-128m-marin-control`, `-conv`, `-embnorm`):

- **Short convolutions** (`short_conv_size: 4`): causal depthwise convolutions
  on full-attention K/V/O and every MLP output, identity-initialized and reset
  at document boundaries (+51,200 params).
- **Embedding RMSNorm** (`embedding_norm: true`): RMSNorm on the embedding
  lookup.

Held-out bits per byte on `kda-code-quality-eval-v1`:

| Evaluation set | Control | Short convolutions | Embedding RMSNorm |
| --- | ---: | ---: | ---: |
| Starred Python | 0.9594 | 0.9441 (-1.6%) | 0.9268 (-3.4%) |
| Other Python | 0.9093 | 0.8986 (-1.2%) | 0.8810 (-3.1%) |
| Starred other code | 0.8174 | 0.8058 (-1.4%) | 0.7841 (-4.1%) |
| Other code | 0.8512 | 0.8399 (-1.3%) | 0.8170 (-4.0%) |
| ClimbMix | 1.1925 | 1.1857 (-0.6%) | 1.1787 (-1.2%) |
| Math | 1.1212 | 1.1116 (-0.9%) | 1.1072 (-1.2%) |

This control differs from the earlier identical run
(`mix-study-v3-20261110-01`) by 0.07–0.32% per set; both options beat that
run-to-run noise on every set.

The training-loss curves diverge, though. Across the five tenths sampled, short
convolutions hold a steady lead (0.015, 0.032, 0.033, 0.038, 0.029 nats), while
embedding RMSNorm's lead shrinks every step (0.232, 0.133, 0.101, 0.087, 0.071
nats). That pattern points to faster early optimization rather than a better
final model, and 200M tokens can't tell them apart.

**Decisions:**

- Keep the 960M foundation run unchanged so its checkpoint lineage stays fixed.
- Test short convolutions in midtraining: splice them in at identity so the
  starting model is unchanged (needs an explicit checkpoint-load path for the
  new parameters).
- Skip embedding RMSNorm for now. It can't be added to a trained checkpoint
  without changing the function, and needs a longer matched run before using it
  in a future pretraining run.

[Run reports, metrics, and common eval](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/marin-ablation-20261111).

## Sizing the run and the loss forecast

**Sizing (November 10).** With its one-block MTP head, the model has 959.8M
training parameters (~104 tokens/param at 100B). At the measured 391K tok/s on
8x H100, 100B tokens takes ~71 hours and ~$22/B tokens at $4/H100-hr — about
$2,200 of the $5,000 foundation allocation.

Stronger base models near this size trained on far more data (Qwen3-0.6B ~36T,
SmolLM2 ~4T). Reaching 4T would cost ~$90,000 and blow past our code supply
(~5B Python, ~45B other permissive GitHub — mostly repeating past ~150B). A
frontier base model from pretraining alone was out of reach, so the plan stops
at 100B, saves budget for midtraining, reasoning SFT, and RL, and targets a
narrow coding category.

**Forecast (November 11).** At 26.4B tokens we fit `loss = c + a * tokens^-b`
to 0.5B-token mean bins over three windows ending at 26B and froze it
([`publication.loss_forecast`](../publication/README.md#loss-forecast)). The
central 5–26B fit is `c = 2.057`, `a = 0.419`, `b = 0.665` (constant LR phase
only, up to the decay at 85B):

| Tokens | Forecast | Measured (0.5B-token mean) |
|---:|---:|---:|
| 26B | 2.105 | 2.098 |
| 50B | 2.088 | 2.085 |
| 75B | 2.081 | 2.064 |
| 85B | 2.079 (range 2.079–2.084) | 2.068 |
| 100B, after decay | not forecast | 1.755 |

- Loss tracked within 0.02 and finished the constant phase slightly below
  forecast, with no stability issues.
- The 59B tokens from 26B to 85B bought ~0.03 in training loss, while the 15B
  decay bought 0.31. The constant phase ran too long — next run should decay
  earlier or use a lower learning rate.
- Held-out BPB followed the same shape on every source.

![Training loss against the forecast, and held-out loss by source](https://huggingface.co/datasets/jfrick/lm-exploration-results/resolve/main/training/results/kda-960m-foundation-20261110-01/forecast/training-loss-forecast.png)

Forecast and chart: `training/results/kda-960m-foundation-20261110-01/forecast/`
in `jfrick/lm-exploration-results`. Run logs: `logs/lineage-*.jsonl` in
[`jfrick/malama-base`](https://huggingface.co/jfrick/malama-base).

## Training and budget

- **Hardware and objective:** 8x 80GB H100 on the shared trainer, frozen
  model/tokenizer, MTP loss sampled at 0.25 (+10.6–10.8% throughput in the
  [matched benchmark](KDA_MIXER_TIMING_STUDY.md#mtp-loss-fraction-november-9);
  quality effect unmeasured).
- **Qualification and ops:** Start with a bounded qualification segment (finite
  losses, per-source held-out eval, checkpoint/resume, source consumption)
  before extending the lineage. Keep periodic checkpoints and numerical evals.
  Every Modal launch uses a server-side timeout <= 24h, zero retries, and an
  independent remote deadline supervisor.
- **Budget:** $5,000 foundation cap. Reconcile actual all-in $/token (prep,
  CPU, memory, storage, eval, checkpoints) before each continuation so the
  post-training reserve stays intact.
