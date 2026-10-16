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
