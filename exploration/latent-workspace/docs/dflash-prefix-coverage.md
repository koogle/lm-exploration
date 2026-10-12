# Training-prefix coverage prototype

Branch: `drafter-data-coverage`. This prototype is isolated from the active
24576/49152-response fits on main. Do not change their package source or resume
those jobs with this branch. No GPU jobs or new teacher datasets have been launched
from this checkout; the existing main-checkout budget ledger remains authoritative.

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
  Article-level cross-split conflicts are rejected in either insertion order;
  exact prompt and cross-source near-duplicate checks remain enforced.
- `load_teacher_cache` uses private, disk-backed tensor mappings for expansion,
  training and diagnostics. It preserves values without eagerly loading all feature
  storage into host RAM. Source files must remain present and immutable. New teacher
  tensors and Python metadata still consume RAM; this is not a claim that arbitrary
  pool creation fits any memory budget.

98 CPU tests pass, including existing exact continuation/warm-start/cache tests,
new article/prefix quarantine tests, and a test that private mapped data cannot
modify the source file even when a shared global mapping default was configured.
Ruff, formatting and diff checks pass. No large-cache or GPU smoke has run for this
branch; no training-quality claim follows from these tests.

## Not integrated or selected yet

The prefix helper is not called by the manifest builder. There is no CLI/config
option or actual expanded manifest yet. Before using it:

1. Integrate a bounded per-article option into prepared-pool generation, preserving
   the exact legacy behavior by default and all existing records/teacher labels.
2. Supply every historical held-out group, register prior manifests, and preserve
   source-group metadata through teacher caching. Report examples and distinct
   source groups separately; keep validation/test identities unchanged.
3. Validate extension/resume compatibility and run a bounded real-model smoke from
   the authoritative ledger after the active matched study finishes.
4. Select the next experiment from completed learning curves/full-fitting audits;
   do not treat this prototype as a decision to train another model.

Reference protocol: [improvement series](dflash-improvement-series.md).
Source-capacity evidence: [pinned article counts](../results/dflash-prose-source-capacity.json).
