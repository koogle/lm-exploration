# ECB policy, infrastructure and project expansion, 19 October 2026

The owner requested continued ECB expansion. A review of 24 exact FOEDB types
identified 1,720 candidate URLs. Twenty-two series are enabled, providing **1,687
unique candidate URLs** in pinned snapshot `1792213219/2sP68GoB`. The enabled URLs
have zero exact overlap with the previously configured types or one another.
Candidate metadata is not a saved-document count.

| Enabled collection | Type | Candidate URLs | HTML | PDF |
| --- | --- | --- | --- | --- |
| weekly_financial_commentaries | 21 | 645 | 645 | 0 |
| monetary_policy_statements | 54 | 298 | 298 | 0 |
| market_infrastructure_news | 183 | 223 | 223 | 0 |
| consumer_expectation_surveys | 298 | 92 | 92 | 0 |
| macroeconomic_projections | 95 | 90 | 31 | 59 |
| monetary_analyst_surveys | 199 | 61 | 0 | 61 |
| monetary_analyst_results | 268 | 43 | 0 | 43 |
| digital_euro_investigation | 279 | 38 | 0 | 38 |
| digital_euro_preparation | 399 | 30 | 0 | 30 |
| digital_euro_rulebook_preparation | 362 | 31 | 0 | 31 |
| digital_euro_rulebook_investigation | 400 | 16 | 0 | 16 |
| digital_euro_project_documents | 471 | 12 | 0 | 12 |
| digital_euro_rulebook_project | 472 | 11 | 0 | 11 |
| digital_euro_progress_investigation | 406 | 5 | 0 | 5 |
| digital_euro_progress_preparation | 408 | 4 | 4 | 0 |
| digital_euro_experiments | 280 | 5 | 0 | 5 |
| ami_seco_reports | 236 | 34 | 0 | 34 |
| balance_of_payments_quality | 149 | 16 | 5 | 11 |
| financial_accounts_quality | 150 | 9 | 6 | 3 |
| monetary_statistics_quality | 207 | 8 | 2 | 6 |
| climate_financial_disclosures | 392 | 8 | 0 | 8 |
| payment_fraud_reports | 145 | 8 | 4 | 4 |

One paced Collector sampled newest, middle and oldest records per series, using
the same 20 MiB limit as production. Of 66 samples in enabled series, 65 fetched
and extracted native full text. None exactly matched the pinned saved-text index.
The first preparation-phase digital-euro progress report returned HTTP 503; it
remains ordinary bounded retrieval work, with no replacement URL or failure reset.
The original PDFs include short public meeting agendas; they are document text,
not catalog metadata, and existing extraction thresholds remain unchanged.
Samples are selected rather than random yield estimates. Passage overlap remains
unmeasured, and exact-content deduplication remains authoritative during ingestion.

Two series are deferred: T2S harmonisation reports (19 URLs) and Euro Money Market
reports (14). In each, only one of the three selected samples retrieved, while two
historical catalog URLs returned HTTP 404. Preserve those observations and repair
links using official evidence before enabling those series. Do not guess endpoints.
All 72 sample outcomes, including these four 404s and the 503, are retained in the
adjacent JSON. Downloaded bodies and bulk inventories remain ignored artifacts.

Both pipeline mappings enable the 22 series at weight 1. Existing weights,
five-second ECB pacing, private storage, single-writer reservations, empty-round
backoff, token stop policy and all budgets remain unchanged. The token-audit limit
remains the separately authorized 96 pairs. Goal and manual apps are redeployed;
the existing controller owns the new ECB campaign, with no independent writer.

The resulting mapping covers 64 of 113 types and 18,927 of 20,162 catalog records
in this **English metadata snapshot**. This describes configured discovery, not
successful retrieval, saved rows or whole-website exhaustion. Remaining types,
other languages, attachments, broken links and extraction failures remain gaps.
Validation: 389 tests plus Ruff lint/format and whitespace checks.
