# European Securities and Markets Authority

Bounded live reconnaissance on 2026-10-16 UTC. Source configuration and individual probe evidence are in the adjacent JSON file. Raw responses are local ignored artifacts, not committed corpus text.

## Findings

- Likely interpretation of the user’s European market agency: European Securities and Markets Authority (not a confirmed clarification).
- Native HTML article text verified on two financial-regulatory news items; linked PDFs include technical reports, manuals and a duplicate press-release representation. PDF bytes and text layers were not fetched or validated.
- The guessed /press-news/esma-news/rss.xml returned HTTP 200 text/html, not RSS; omit as a feed seed.
- The list and ?page=1 return distinct HTML with article links. Full archive page count remains unmeasured.
- Article HTML includes repeated full and side-teaser article blocks. Extract the primary article body and deduplicate; stripping all page text will retain related-news and navigation noise.
- Robots excludes search, administration and user paths; the sampled press-news and publications routes are not disallowed. English samples only; multilingual technical reports need later enumeration.

## Tested requests

| URL | UTC time | HTTP | Content type |
| --- | --- | --- | --- |
| https://www.esma.europa.eu/robots.txt | 2026-10-16T08:59:43.272577+00:00 | 200 | text/plain; charset=UTF-8 |
| https://www.esma.europa.eu/press-news/esma-news | 2026-10-16T08:59:43.586699+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.esma.europa.eu/press-news/esma-news/rss.xml | 2026-10-16T08:59:43.791513+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.esma.europa.eu/publications-and-data | 2026-10-16T08:59:44.546496+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.esma.europa.eu/press-news/esma-news/one-year-go-t1-settlement-regulatory-framework-complete-now-lets-make-it | 2026-10-16T09:00:30.417129+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.esma.europa.eu/press-news/esma-news/esma-updates-its-manual-transparency-new-guidance-otc-derivatives-and-tick | 2026-10-16T09:00:30.897431+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.esma.europa.eu/press-news/esma-news?page=1 | 2026-10-16T09:00:31.354022+00:00 | 200 | text/html; charset=UTF-8 |

## Collection recommendation

Use the `recommended_source` configuration in the JSON report. The feed/listing is a discovery surface; only fetched article bodies count as training text. Preserve publication date, source URL, language, retrieval timestamp, format and content hash. Keep linked reports distinct from press summaries; deduplicate HTML/PDF copies.

No total document count or token estimate is established by this probe. Historical coverage, multilingual coverage and incremental-update reliability remain to be measured. Recheck current robots instructions before each collection run.
