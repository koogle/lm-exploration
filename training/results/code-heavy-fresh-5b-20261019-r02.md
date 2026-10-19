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

The persisted new allocation was verified: LibreTexts now has an 11,189,950-target
phase budget backed by 5,874,724 fresh unique targets plus reserve. Its capacity
gate passes. All 197,297,522 previously prepared **fresh** targets were retained;
these are prepared targets, not training updates. Every non-book budget remains
unchanged and the book budgets still sum exactly to 714,285,715.

Training completed October 19 at **21:53 Cyprus time** (18:53 UTC):
4,999,971,081 additional loss-bearing targets over 66,065 updates, reaching
10,878,562,619 lifetime targets. The remaining 28,919 targets did not fill the
next optimizer batch and were left unused. Coding contributed 2,999,962,024
exposures (approximately 60%). The main segment took 5h 54m; start/restore
qualification also completed successfully. Stop reason: token limit; no data error.

Final private checkpoint:
`simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`
(1,493,001,498 bytes), verified present on the training volume. The adjacent JSON
records complete source exposure counts, final counters and checkpoint metadata.

Expanded validation completed: 7,620,187 matched targets, all 14 categories, using
the existing 2,782-family-selection sample and paired bootstrap protocol. See
[aggregate results](code-heavy-fresh-validation-20261019.json). Coding validation
loss improved from 1.559 (original 5B model) and 3.994 (EU parent) to **1.421**.
All 12 original sources improved relative to the EU parent; 11 non-coding original
sources remain slightly worse than the original 5B checkpoint. ECB loss is 2.494
and EMA loss is 2.379: both retain small gains over the original 5B model but have
lost most of the EU parent's specialist gains (2.065 and 1.766, respectively).
These are held-out loss results, not evidence of generation quality or factual
correctness. Existing split/near-duplicate/quality audit limitations still apply.

Training and validation are complete. No further training or public Hugging Face
upload was started; checkpoints and private samples remain on Modal.
