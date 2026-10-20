# Python 124M / 5B execution record

**Status: CPU preparation; no GPU training or model publication.**

[Draft PR #30](https://github.com/koogle/lm-exploration/pull/30) remains stacked on
#27. It incorporates the tested Muon implementation from #24, at
`1b9a7379e361d92f7e2ace0b15758a7e6eeed2f5`. The requested six-category mixture and
5,000,000,000 nonpadding-target budget remain unchanged. All code is Python;
initialization is fresh when training is eventually qualified.

## Executed work

GitHub, Modal and Hugging Face authentication succeeded. Both private code
collections were inspected: they contain complete source snapshots as well as
patch records. Preparation reads complete blobs from their 58 stored Git
repositories, pins commits, and prioritizes more recent snapshots. It does not
train on patch strings or execute repository code. Not every stored snapshot is
recent; commit dates remain in the private provenance.

The initial per-file Git reader was replaced with a batched blob reader, retaining
files marked `export-ignore`. Completed repositories were resumed rather than
recollected. Five repository license checks were unresolved and excluded.

The completed repository-only census contains **109,455,272 unique training
targets** across implementation, tests/examples and documentation. These counts
include one EOS target per document and exclude validation, filtered files, exact
duplicates and per-repository caps. They are **not** a 5B-capable corpus. Further
cross-source held-out exclusions can reduce them during supplemental preparation.
See [capacity table](repositories-capacity.md) and [machine-readable counts](repositories-capacity.json).

- [Initial repository preparation](https://modal.com/apps/radiantai/main/ap-vvHeCt6S23k42AiZKcL0wt): stopped to replace inefficient file-at-a-time reads.
- [Resumed, completed repository preparation](https://modal.com/apps/radiantai/main/ap-qCWLuFTlBWJHHHv7gP4dfY).
- [Detached supplemental preparation](https://modal.com/apps/radiantai/main/ap-SN8iRtna31TPxCgvyq6WVy): CPU-only, bounded continuation with persisted cursors.

The Stack, The Stack Dedup and StarCoderData authenticated file probes returned
`GatedRepoError`: the account is not authorized to retrieve their Python shards.
Existing pinned CodeParrot pools and supporting-text collections are being
inspected instead. A dataset's advertised size is not usable Python capacity.

## Checks and limits

Six corpus tests and eight configuration/resume-policy tests passed locally.
Ruff and whitespace checks passed. Prior eight-H100 compiled Muon qualification,
optimizer restore and full-model continuation results are retained in the
[original measured report](../muon-124m-100mt-20261019.md); those are prior-run
results, not GPU measurements from this experiment.

The new preparation preserves source text, comments, indentation and docstrings;
assigns each accepted file to one category; rejects generated/vendored/minified
and heavily repetitive files; and excludes files rejected by the Python 3.12 parser explicitly. Valid syntax introduced in later Python versions may therefore be excluded.
Path-based task classification and root-license detection are heuristics, not an
exhaustive per-file license or semantic-quality audit. Exact deduplication does
not remove all near clones. Larger files and some fixtures are conservatively
excluded. Supporting sources may be older. No benchmark-clean claim is made.

Private source text, repository-level notices and detailed provenance remain on
the private Modal data volume. Only aggregate capacity is committed here.

## Pending, not claimed

Corpus sufficiency, a frozen representative validation set, new-corpus GPU
qualification, tiny-set fitting/completion, training/generation logit agreement,
full 5B training, isolated executable evaluation, checkpoint publication, curves,
unselected generations and final Hub hash verification remain pending. No new
GPU optimizer step, training checkpoint, model repository or training cost is
claimed. A queued controller or preparation worker is not a training run.
