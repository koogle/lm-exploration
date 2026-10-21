# European Central Bank

Bounded live reconnaissance on 2026-10-16 UTC. Source configuration and individual probe evidence are in the adjacent JSON file. Raw responses are local ignored artifacts, not committed corpus text.

## Findings

- RSS is metadata and URLs, not full article text. Fifteen current items observed; do not extrapolate archive size from this window.
- Native HTML full text verified for speech, interview, survey press release and Governing Council decisions. Meeting account sample returned 503; preserve failure and retry later.
- Robots requests Crawl-delay: 5 and disallows *_content.<language>.html fragments and several shared video paths. Initial independent seed probes were simultaneous; follow-up article requests were spaced five seconds. Production must enforce per-host scheduling.
- English sampled only. Language variants and full historical pagination need separate inventory. Feed URLs contain a double slash after the host.
- Banking supervision links point to www.bankingsupervision.europa.eu; this separate host is outside this bounded adapter. European Investment Bank would also require its own source.

## Tested requests

| URL | UTC time | HTTP | Content type |
| --- | --- | --- | --- |
| https://www.ecb.europa.eu/robots.txt | 2026-10-16T08:59:43.199183+00:00 | 200 | text/plain; charset=UTF-8 |
| https://www.ecb.europa.eu/rss/press.html | 2026-10-16T08:59:43.204145+00:00 | 200 | application/rss+xml |
| https://www.ecb.europa.eu/rss/press.xml | 2026-10-16T08:59:43.462743+00:00 | 200 | text/xml |
| https://www.ecb.europa.eu/press/html/index.en.html | 2026-10-16T08:59:43.929209+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.ecb.europa.eu//press/key/date/2026/html/ecb.sp261014~e4ca33588e.en.html | 2026-10-16T09:00:13.644998+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.ecb.europa.eu//press/accounts/2026/html/ecb.mg261008~a10153d090.en.html | 2026-10-16T09:00:19.109495+00:00 | 503 | text/plain |
| https://www.ecb.europa.eu//press/inter/date/2026/html/ecb.in261008~3184e7d0d0.en.html | 2026-10-16T09:00:24.971309+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.ecb.europa.eu//press/pr/date/2026/html/ecb.pr261007~6447350434.en.html | 2026-10-16T09:00:54.441643+00:00 | 200 | text/html; charset=UTF-8 |
| https://www.ecb.europa.eu//press/govcdec/otherdec/2026/html/ecb.gc261002~54c6b5672b.en.html | 2026-10-16T09:00:59.641752+00:00 | 200 | text/html; charset=UTF-8 |

## Collection recommendation

Use the `recommended_source` configuration in the JSON report. The feed/listing is a discovery surface; only fetched article bodies count as training text. Preserve publication date, source URL, language, retrieval timestamp, format and content hash. Keep linked reports distinct from press summaries; deduplicate HTML/PDF copies.

No total document count or token estimate is established by this probe. Historical coverage, multilingual coverage and incremental-update reliability remain to be measured. Recheck current robots instructions before each collection run.
