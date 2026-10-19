# Fresh 5B continuation: replacement launch

Launched October 19, 2026 at 09:44 UTC (12:44 Cyprus), run
`simple-llm-code60-fresh-5b-20261019-r02`, Modal app `simple-llm-code-heavy-5b`.
Runtime source: `654e980`. Controller and exact provenance are in the adjacent JSON.

The previous attempt stopped safely when LibreTexts exhausted. The user requested
fixing and continuing. Keep the full 5B target, coding at 60%, and every non-book
source budget. Keep the total books budget at 714,285,715 targets. Cap exhausted
book collections using only measured fresh capacity with a 5% reserve and two-pass
limit, then redistribute their shortfall among the remaining original book
collections. This changes source weights within books, not category budgets.
No additional unfiltered datasets or loosened filters are introduced.

Reuse the committed fresh staging data, prior-hash exclusion ledger, and cursors.
Initialization now reuses that verified ledger without hashing old multi-GB corpora
again. The final immutable `code60-5b-books-v2` manifest records original and final
allocations and verifies zero prior-document hash overlap in its training set.
The original failed run and immutable snapshots remain intact.

Twelve CPU tests pass, including repeated book exhaustion, exact integer budgets,
non-book budget invariance, zero-capacity source omission, fresh-process restart,
valid training-loader output, hash exclusion, conflict accounting, and checkpoint
migration. Ruff and whitespace checks pass.

The new controller is preparing data. New-data GPU qualification has not yet
completed, and no training tokens are claimed. After fresh capacity passes, the
workflow automatically qualifies start/restore (two updates each), then trains
5B targets with eight H100s, 8K context, cropping, cached RoPE, compilation and
microbatch two. Checkpoints remain private; completion queues the existing 7.62M
expanded validation sample against both earlier checkpoints.

This is a launch record, not a live completion assertion. Read the deployed
`inspect` function or run-volume workflow and staging progress for current state.
