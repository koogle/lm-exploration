# ECB collection enumeration — 2026-10-16

The current ECB publication search uses **FOEDB**, a public versioned JSON
inventory. Its own `foedb.min.js` supplies the schema and URL construction rules.
This is a substantially better historical discovery surface than the 15-item RSS
window. Inventory records contain document links and metadata, not full text.

## Verified process

1. Fetch `/foedb/dbs/foedb/publications.en/versions.json` and pin its first
   version/hash for the entire enumeration.
2. Fetch `<version>/<hash>/metadata.json` for columns, chunk sizes and total.
3. Fetch `indexes/type/index.json`. The array position is the type's index ID.
4. Fetch `indexes/type/<index-id>/chunk_<n>.json` for each relevant type. These
   contain sorted record positions, not publication IDs.
5. For each position, fetch `data/<group>/chunk_<n>.json`. Records are flattened
   arrays with the metadata header width. Position / chunk_size gives chunk;
   chunk / chunk_group_size gives group.
6. Read `documentTypes` as alternative full-document representations. Prefer
   HTML per language, retain explicitly listed translations, normalize URL
   duplicate slashes and retain the original publication ID separately.
7. Download and extract each selected URL using the shared harness; none of the
   inventory abstracts are substituted for full documents.

Observed inventory snapshot: `1792148434/maNnqDza`; **20,160 publication records**.
The first data chunk contains recent 2026 publications. Last chunk (80) includes
1998 and older material, with its earliest timestamp in March 1992. This does not
prove every intervening document is covered. Full historical download was not run.

| Code collection | FOEDB type IDs | Inventory publication records |
| --- | --- | ---: |
| speeches | 19 | 3,079 |
| press_releases | 1, 92 | 2,557 |
| interviews | 27 | 617 |
| meeting_accounts | 20 | 97 |
| decisions | 55 | 248 |

These sum to **6,598 records**, before language variants or representation filtering.
Working papers and many other series are also visible but outside these five
initial adapters. English inventory metadata can explicitly list non-English
PDF variants. Other language databases have not been inspected. URLs without an
explicit language suffix are currently skipped; linked children/attachments are
not recursively collected. Reported source counts are not extracted-text counts.

## Negative findings and bounds

- `/press/key/html/index.en.html`: HTTP 200 but the body is a JavaScript search
  container, not an archive with all speech links.
- `/press/pr/date/2025/html/index.en.html`: HTTP 404; do not invent yearly routes.
- `/press/accounts/html/index.en.html`: HTTP 200.
- `/sitemap.xml`: HTTP 200, 480 canonical navigation URLs, not article coverage.
- `robots.txt`: HTTP 200; five-second crawl delay and `_content` fragments excluded.
- JavaScript assets, version manifest, metadata, type index, first speech index,
  first data chunk and final data chunk all returned HTTP 200.

Thirteen bounded live requests, at least five seconds between requests to this
origin. Raw responses retained under ignored `artifacts/ecb-enumeration/`.
An offline replay of the live responses successfully enumerated the newest five
speech documents. Four synthetic tests check type-index decoding/chunk groups,
HTML preference and translations, stable IDs across backfills, limit completeness,
unsafe hosts, excluded content fragments, and propagation of network failure.

Enumeration is newest-first and bounded by output URLs. To backfill beyond a
previous bound, increase that bound; the shared durable harness skips already
completed document IDs. Each run pins its own inventory snapshot. The adapter
records snapshot version/hash, requested inventory URLs, count, coverage
limitations and whether this snapshot/type slice was exhausted. It does not claim
completeness of the institution. It does not silently downgrade API failures to
an empty successful enumeration.
