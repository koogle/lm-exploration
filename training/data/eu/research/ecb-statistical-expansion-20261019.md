# ECB statistical archive expansion, 19 October 2026

The owner requested continued ECB collection after the survey/T2S backlog finished.
Twelve exact FOEDB types provide 2,239 unique candidate URLs in the pinned English
snapshot `1792213219/2sP68GoB`. They have zero exact URL overlap with the previously
configured types or one another. These are metadata candidates, not saved documents.
Publication records and representation URLs differ because explicit translations
are retained and duplicate same-language representations collapse.

| Collection | Type | Candidate URLs | HTML | PDF |
| --- | --- | --- | --- | --- |
| weekly_financial_statements | 101 | 801 | 472 | 329 |
| monetary_developments | 10 | 358 | 97 | 261 |
| mfi_interest_rate_statistics | 8 | 298 | 98 | 200 |
| securities_issues_statistics | 9 | 268 | 46 | 222 |
| balance_of_payments_monthly | 53 | 166 | 143 | 23 |
| investment_fund_statistics | 12 | 103 | 22 | 81 |
| financial_vehicle_statistics | 13 | 50 | 23 | 27 |
| balance_of_payments_quarterly | 56 | 57 | 49 | 8 |
| institutional_sector_accounts | 90 | 55 | 33 | 22 |
| insurance_corporation_statistics | 97 | 27 | 22 | 5 |
| pension_fund_statistics | 237 | 15 | 15 | 0 |
| money_market_statistics | 113 | 41 | 41 | 0 |

One paced Collector sampled newest, middle and oldest records from each series.
All 36 fetched and parsed using the existing native HTML/PDF extraction path.
Raw hashes, extraction sizes, titles and opening text are recorded in the adjacent
JSON; bodies remain ignored artifacts. Samples are selected, not a random yield
estimate. Numeric financial statements remain original text, not catalog metadata
or generated prose. Passage overlap is unmeasured; exact-content deduplication
continues during collection.

Both mappings add these series at weight 1, preserving existing weights, five-second
ECB pacing, private storage, single-writer reservations and all operational limits.
The controller recognizes the new inventory work even though its previous saved
remaining count was zero. The goal and manual apps are redeployed. Empty-round
backoff and source cooldowns remain. The recovery schedule remains disabled.

Validation: 361 tests passed; Ruff lint/format and whitespace checks passed. Tests
exercise each exact type, native HTML preference and distinct language variants.
This expansion does not establish whole-website exhaustion; excluded whole issues,
known sections, other languages, attachments and unresolved extraction failures
remain coverage gaps.
