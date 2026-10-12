# Training-prefix coverage prototype

Originally developed on `drafter-data-coverage`; now integrated on main for the
fresh grid. The prior fits completed before integration. Prefix/cache safeguards
passed local tests and real-model preparation/training checks; full teacher
preparation is active under the Modal controller. The remote study ledger is
authoritative while its ownership marker exists. See [current work](../OPEN_WORK.md).

The pinned WikiText audit found 23090 training article groups before filtering.
One prompt per article therefore caps a balanced pool at 69270 responses. Extra
prefixes could increase distinct training contexts while preserving article-level
holdouts; they are not additional independent documents.

## Implemented and tested locally

- `dflash_prefixes.training_prose_prefixes` samples deterministic nonoverlapping
  prefixes from articles assigned to training. It honors explicit historical
  holdout exclusions and a per-article limit that includes existing offsets.
- Additional records have a distinct example `group`, plus `source_group` retaining
  the original article identity. Their example IDs do not determine split membership.
- `PromptRegistry(allow_training_prefixes=True)` allows only correctly identified
  additional training prose prefixes to reuse a source. Default behavior is unchanged.
  Articles assigned to validation/test cannot be relabeled as training even before
  registration. Article-level cross-split conflicts are rejected in either insertion order;
  exact prompt and cross-source near-duplicate checks remain enforced.
- `load_teacher_cache` uses private, disk-backed tensor mappings for expansion,
  training and diagnostics. It preserves values without eagerly loading all feature
  storage into host RAM. Source files must remain present and immutable. New teacher
  tensors and Python metadata still consume RAM; this is not a claim that arbitrary
  pool creation fits any memory budget.

108 CPU tests pass, including existing exact continuation/warm-start/cache tests,
new article/prefix quarantine tests, and a test that private mapped data cannot
modify the source file even when a shared global mapping default was configured.
Ruff, formatting and diff checks pass. No large-cache or GPU smoke has run for this
branch; no training-quality claim follows from these tests.

## Manifest integration, locally verified

`DraftConfig.prose_prefixes_per_article` now enables a bounded 1–8 prefix cap
(default 1). Diverse pool preparation calls the helper for training prose only,
registers all prior manifests, retains all existing records, and records distinct
source groups separately from example counts. Articles with unknown historical
prefix offsets are skipped for expansion. Validation/test examples are unchanged.
Extension cannot decrease the retained cap; exact training continuation checks it.

Integration tests exercise the actual manifest builder with controlled source rows:
opt-in expansion, unchanged legacy cap, historical holdout rejection, unknown-offset
rejection, and preservation of all existing records. Cache tests still verify that
old teacher labels/features are unchanged and test is not generated.

No GPU smoke or full-size manifest has run. Before using this branch:

1. Reconcile and merge only after current fits finish; keep active audit provenance.
2. Run bounded real-model preparation smoke from the authoritative main ledger,
   including extension of a prefix-enabled pool and unchanged held-out cache hashes.
3. Validate actual full-size prompt capacity, filtering, memory and preparation time.
4. Select the next experiment using completed curves/full-fitting audits. Extra
   prefixes remain correlated contexts from existing articles, not new documents.

Reference protocol: [improvement series](dflash-improvement-series.md).
Source-capacity evidence: [pinned article counts](../results/dflash-prose-source-capacity.json).
