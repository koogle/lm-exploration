# ECB report-series expansion, 18 October 2026

The four exact FOEDB publication types below each passed one native full-text
sample. Both source mappings now include them with scheduling weight 3, using
the existing bounded ECB writer and five-second pacing. Organization-wide exact
text deduplication continues to apply. No operational cap was increased.

| Collection | Type | Catalog publication records | Sample characters | Format |
| --- | ---: | ---: | ---: | --- |
| Macroprudential Bulletin articles | 96 | 85 | 50,381 | HTML |
| Financial integration and structure reports | 140 | 16 | 147,767 | HTML |
| International role of the euro | 143 | 25 | 88,435 | HTML |
| TARGET annual reports | 142 | 24 | 99,532 | HTML |

These are 150 catalog publication records, not saved rows or verified tokens.
Explicit translations can produce more than one candidate URL per record.
The sample opening text matches the catalog title and contains substantive
article/report prose. The integration report includes its preface and report
sections; the TARGET sample is the 2023 report. One sample per series does not
establish that every historical document is retrievable or fully extracted.

[Machine-readable evidence](ecb-report-series-20261018.json) records exact URLs,
FOEDB snapshot receipts, live document timestamps, raw and extracted text hashes,
and sample opening text. Existing local FOEDB response caches reduced requests;
cache entries retain their original retrieval times. The existing adapter still
prefers native HTML to a same-language PDF and retains explicit translations.
Passage overlap with other publications has not been measured.

The shared type/representation regression runs for all four additions, including
same-language format preference and distinct French PDF retention. Validation:
287 tests passed, Ruff lint/format and whitespace checks passed. Deployment
receipts and actual subsequent worker coverage are recorded in the continuation
report. Newly deployed code does not establish that an already running worker
has adopted the new mapping.
