# Historical ECB Bulletin links — 2026-10-16

The running collector reported historical article URLs returning HTTP 404 under
`/press/economic-bulletin/articles/2014/pdf/`. Two exact standalone replacements
were verified on the official ECB host. This is not evidence that all failed URLs
can be repaired with a global prefix substitution.

| Inventory filename | Verified standalone URL | HTTP | Bytes | PDF pages |
| --- | --- | ---: | ---: | ---: |
| art1_mb201402en_pp77-96en.pdf | https://www.ecb.europa.eu/pub/pdf/other/art1_mb201402en_pp77-96en.pdf | 200 | 402,082 | 20 |
| art3_mb201401en_pp83-97en.pdf | https://www.ecb.europa.eu/pub/pdf/other/art3_mb201401en_pp83-97en.pdf | 200 | 284,833 | 15 |

The first article is **Extensions to the models for assessing money and credit**,
February 2014, printed pages 77–96. The second is **Medium-term prospects for
China’s economy and the internationalisation of the renminbi**, January 2014,
printed pages 83–97. Both response bodies were parsed as PDFs and first-page titles,
dates, and numbering inspected. Neither response redirected to a generic page.

## Independent full-issue cross-check

The official sitemap links the [Monthly Bulletin archive](https://www.ecb.europa.eu/press/economic-bulletin/mb/html/index.en.html).
This archive returned HTTP 200 and directly links each issue from 1999–2014,
including [February 2014](https://www.ecb.europa.eu/pub/pdf/mobu/mb201402en.pdf).
The February issue returned HTTP 200, 5,718,425 bytes.

The missing February article is exactly physical PDF pages 78–97 (printed 77–96).
Normalized extracted text from those 20 issue pages **exactly equals** text from
the verified standalone article: 74,162 normalized characters, SHA-256
`b4d618493fbd9a3dffdf1a3de591fddb19f45a768e0d3d0e72de7f69fd1f04da`.
This independently confirms document identity rather than relying on filename
similarity or merely an HTTP-success response. No whole-issue substitution is used.

## Implementation and limits

`adapters.ecb.alternate_document_urls(record)` returns a list containing the
verified standalone URL for those two exact mappings, and an empty list otherwise.
The parent harness calls it only after primary HTTP 404, retains the original
inventory record and item key, and records the primary and fallback URLs in fetch
provenance. No input mutation, document identity change, or whole-issue substitution
occurs. The running worker was not edited, interrupted, redeployed, or refreshed here.

The `/pub/pdf/other/` locations were investigated as hypotheses suggested by the
existing ECB publication inventory's legacy PDF structure, then accepted only
after successful downloads and document-content validation. Other failed links
remain unresolved until similarly checked. The full-issue archive provides a
possible future page-aware fallback, but printed/physical page offsets and article
boundaries must be validated per issue; adding the entire issue as an article
would be incorrect and duplicate unrelated text.

Five ECB requests during this investigation: robots, the sitemap-linked archive,
February issue PDF, and the two standalone PDFs. Requests were separated by at
least five seconds and the current robots rules were checked. Two external search
probes supplied no useful evidence and are not relied upon. Raw responses are in
ignored `artifacts/ecb-historical-links/`; only code and findings are versioned.
Six adapter tests and lint pass, including exact mapping, retained identity,
nonmutation, and leaving unverified links unchanged.

## Follow-up failure audit and expansion — 2026-10-17

Read-only HF snapshot `eb3cfd398b0a5947724658acf33402d27e646427` contained
364 failures: 358 HTTP 404, four response-size exclusions and two short-text
exclusions. Matching the saved article inventory identified 341 failed article
URLs. This is a measured repair opportunity, not 341 verified replacements.

A bounded follow-up checked twelve 2014 article replacements at the already
verified legacy `/pub/pdf/other/` location. Eleven returned HTTP 200 PDFs whose
extracted text contains the corresponding exact inventory title (case and
punctuation normalized) near the beginning. Their bodies total **174 PDF pages,
621,746 extracted characters, 97,120 whitespace words and 8,571,472 bytes**.
These are downloadable repair candidates, not additional rows already published.
The adapter now contains 13 exact replacements including the original two.

| Verified additional filename | Pages | Extracted words | UTC observation |
| --- | ---: | ---: | --- |
| art1_mb201412.en.pdf | 13 | 6,918 | 2026-10-17T04:26:57.732532+00:00 |
| art2_mb201411_pp91-105.en.pdf | 15 | 8,591 | 2026-10-17T04:27:04.611542+00:00 |
| art2_mb201410_pp69-82.en.pdf | 14 | 7,838 | 2026-10-17T04:27:16.596859+00:00 |
| art1_mb201410_pp49-68.en.pdf | 20 | 9,461 | 2026-10-17T04:27:22.401554+00:00 |
| art1_mb201409.en.pdf | 15 | 8,557 | 2026-10-17T04:27:28.693845+00:00 |
| art2_mb201408_pp83-97en.pdf | 15 | 8,550 | 2026-10-17T04:27:34.185123+00:00 |
| art1_mb201408_pp65-82en.pdf | 18 | 10,447 | 2026-10-17T04:27:39.619877+00:00 |
| art3_mb201407_pp99-114en.pdf | 16 | 10,339 | 2026-10-17T04:27:46.416342+00:00 |
| art2_mb201407_pp79-97en.pdf | 19 | 10,399 | 2026-10-17T04:27:52.187060+00:00 |
| art1_mb201407_pp63-77en.pdf | 15 | 8,280 | 2026-10-17T04:27:58.255817+00:00 |
| art1_mb201406en_pp101-114en.pdf | 14 | 7,740 | 2026-10-17T04:28:04.678961+00:00 |

The twelfth probe, `art1_mb201411_pp79-90.en.pdf`, returned HTTP 503 and was
**not added**. No inference of successful repair is made for it. Regression
coverage explicitly checks that this unsuccessful probe remains unmapped.

Thirteen ECB requests: current robots plus twelve candidate PDFs, with at least
five seconds between requests and robots allowance checked before download.
HF progress and saved inventory reads were read-only; active Modal writers were
not changed. Evidence and raw PDFs remain under ignored
`artifacts/ecb-historical-links/verified-expansion.json` and `expanded-*.pdf`.
Future workers must retain the primary inventory URL and stable item identity,
try these alternatives only after primary HTTP 404, and record fetch provenance.
No whole-issue substitution, generic URL rewrite, or HF write was performed.

## Runtime-validated inference — 2026-10-17

The adapter now additionally proposes exactly one same-basename legacy URL for
four narrowly defined Monthly Bulletin article filename families. Preconditions:
exact HTTPS ECB host, `economic_bulletin_articles` collection, original
`/press/economic-bulletin/articles/YYYY/pdf/` path, 1999–2014 year matching the
filename, month 01–12, and positive ordered page ranges at most 999 where supplied.
Whole-issue names, query strings, nested paths, newer Economic Bulletin names and
focus/box paths do not match. The 13 individually verified mappings stay trusted.

Offline matching covers **293 of the 341 recorded article 404s** (293 of all 358
HTTP 404s). No basename collision or different-title collision was found in the
retained failure inventory. Coverage is a count of proposed recovery attempts,
not successful downloads. The previously unsuccessful 503 probe is now eligible
only as an inferred candidate; it has not become a trusted mapping.

The parent must invoke `validate_alternate(original_candidate, response_bytes)`
**after fallback download and before normal extraction**. For inferred candidates
this requires a PDF body and the inventory title as a contiguous normalized
substring in extracted text from the first two PDF pages. Normalization removes
punctuation/whitespace, case differences and decomposable accents. Missing titles,
fewer than three alphabetic words, or fewer than twelve normalized characters
are rejected. An image-only title, OCR errors or materially abbreviated title
can cause conservative false negatives; these remain failures for later review.
No fuzzy title acceptance is implemented. Exact previously verified mappings are
exempt from this additional check because their identity was already established.

No live source requests were made for this generalization. Synthetic tests cover
all four filename families, malformed dates/pages, whole-issue rejection, title
mismatch, match only on page three, non-PDF bodies, short titles, and retaining the
previous exact mappings. The pipeline integration remains owned by the parent.
