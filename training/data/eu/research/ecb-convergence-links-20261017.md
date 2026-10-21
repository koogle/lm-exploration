# ECB convergence report link recovery — 2026-10-17

Three exact historical URLs from the cached official FOEDB inventory have verified
standalone replacements. These are bounded same-basename legacy-path hypotheses,
**not links obtained from an index**: the current official convergence index
returned HTTP 503 at 10:38 UTC. That negative result is retained in the JSON report.

| Original basename under `/press/other-publications/convergence/pdf/` | Verified replacement | PDF identity | Pages | Extracted characters |
| --- | --- | --- | ---: | ---: |
| `cr2006en.pdf` | https://www.ecb.europa.eu/pub/pdf/conrep/cr2006en.pdf | Convergence Report, May 2006 | 88 | 246,962 |
| `cr2000en.pdf` | https://www.ecb.europa.eu/pub/pdf/conrep/cr2000en.pdf | Convergence Report, 2000 | 86 | 200,065 |
| `cr200705en.pdf` | https://www.ecb.europa.eu/pub/pdf/conrep/cr200705en.pdf | Convergence Report, May 2007 | 91 | 266,451 |

All three returned HTTP 200 with `application/pdf`, without redirects. Full PDFs
were parsed using pypdf; their first pages explicitly identify the report title,
year and, for 2006/2007, May. These match the original inventory titles and
publication dates. The 2000 report includes a statistical cutoff in April 2000,
consistent with its May publication. Extracted text is English.

The machine-readable report records exact original/replacement pairs, inventory
identities, retrieval timestamps, byte counts, SHA-256 hashes and first-page
identity evidence. Runtime integration should require the exact original URL,
`convergence_reports` collection and verified raw-body SHA-256 after fallback
fetch; retain the original candidate identity plus both URLs in provenance.
Do not apply a generic prefix rewrite or trust other filenames based on these
three successes. A changed PDF hash should require renewed verification.

The shared Collector checked robots and enforced at least five seconds between
requests per host. There were six requests including two robots reads, the failed
index and three successful PDFs. Raw responses and detailed logs are ignored in
`artifacts/ecb-convergence-links-current/` and
`artifacts/ecb-convergence-links-probes/`. No adapter, corpus checkpoint or active
worker was changed by this investigation. These 713,478 characters are verified
recoverable content, not a claim of additional rows already uploaded.
