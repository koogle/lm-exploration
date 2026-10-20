# ECB additional statistics and SESFOD coverage

On 20 October 2026, ECB's configured backlog had reached zero supported
retry-eligible candidates, with 19,305 saved rows at source revision
`884aa8409f481203b015056a67693575ddd999ed`. This is configured coverage,
not whole-website exhaustion.

The cached official English FOEDB snapshot `1792213219/2sP68GoB` provides
three additional exact types: insurance corporation and pension fund statistics
(11: 24 PDF URLs), payment statistics (14: 21 URLs, 12 HTML and nine PDF), and
credit terms and conditions surveys / SESFOD (454: 24 PDF URLs). The **69
distinct URLs** have no exact URL overlap against previously configured types
or one another. Metadata candidates are not saved documents. Native language
metadata is retained: payment statistics includes four German versions;
some older PDFs have no filename language marker and remain `und`.

Newest/middle/oldest samples from each series were fetched with the normal
five-second source pacing and 20 MiB bound while no ECB collector was running.
All **nine samples** yielded native text above the unchanged 200-character
threshold (7,087–77,189 normalized characters). None exactly matches the pinned
saved-text index. Passage overlap has not been measured. No endpoints were
guessed or extraction/failure thresholds weakened.

Both mappings add these three series with the existing default weight of one.
They provide real new work for the durable raw-document queue rollout.
The adjacent JSON retains tested URLs, retrieval timestamps, hashes, extracted
openings, sample outcomes, overlap checks and the pinned saved-text comparison.
