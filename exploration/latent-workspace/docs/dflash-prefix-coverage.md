# Training-prefix coverage

Originally developed on `drafter-data-coverage`, now integrated on main for the
[fresh grid](dflash-fresh-grid.md). The earlier fits completed before integration.
See [current work](../OPEN_WORK.md) for execution and authoritative budget ownership.

The pinned WikiText audit found 23090 eligible training articles before filtering.
One prompt per article caps a balanced three-domain pool at 69270 responses.
Additional training-only prefixes increase context coverage while retaining
article-level holdouts; they are correlated examples, not independent documents.

## Implemented safeguards

- Deterministic, nonoverlapping prefixes retain `source_group` article identity
  while assigning distinct example IDs. Historical held-out articles are excluded.
- `PromptRegistry` permits source reuse only for valid additional training prose
  prefixes; cross-split conflicts, exact prompts and near duplicates remain rejected.
- `prose_prefixes_per_article` is bounded at 1–8, defaults to legacy behavior, and
  counts existing offsets. Articles with unknown prior offsets are not expanded.
- Preparation retains prior records, labels and original split identities. It
  reports examples separately from distinct source groups. Extension cannot reduce
  its prior cap; exact training continuation checks configuration identity.
- Private mapped cache reads avoid eagerly loading dense historical feature storage.
  Source files must remain immutable and present; new tensors/metadata still use RAM.
- The fresh-grid teacher cache stores training tokens and computes frozen features
  online, avoiding a dense hundreds-of-GiB expanded feature cache.

## Executed validation and remaining work

112 local tests pass, including actual manifest extension, historical holdout
quarantine, malformed prefix rejection, private-cache immutability, causal online
features, bounded teacher resumption and ledger ownership. Real-model preparation
and short/long online-feature training profiles passed. Profiles do not establish
learning quality.

The actual 216k training manifest is prepared: 72k examples/domain, with 18k prose
articles and 72k source groups each for code and translation. Teacher generation
and the five full fresh fits remain in progress/pending under the Modal controller.
Before interpreting results, verify usable counts after filtering, response lengths,
old-token prefix retention and common held-out identities. Keep test sealed and
report the correlation between prefixes rather than labeling all 216k examples
independent sources.

References: [source-capacity audit](../results/dflash-prose-source-capacity.json),
[improvement series](dflash-improvement-series.md),
[fresh-grid protocol](dflash-fresh-grid.md).
