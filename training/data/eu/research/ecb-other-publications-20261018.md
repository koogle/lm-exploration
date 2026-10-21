# ECB Other publication catalog, 18 October 2026

The exact FOEDB type 23, officially named “ECB Other publication”, contains 772
catalog publication records in the inspected snapshot. It includes technical
manuals, agreements and conference proceedings. The adapter now exposes this
single type as `other_publications`; both source mappings assign weight 3 using
the same bounded ECB writer and source pacing.

Three selected, non-random PDF samples passed native pypdf extraction:

| Catalog title | Extracted characters |
| --- | ---: |
| MoU on information sharing in support of non-bank payment service providers access to central bank operated payment systems | 34,804 |
| Government finance statistics guide | 182,361 |
| Financial statistics for a global economy — Proceedings of the 3rd ECB conference on statistics | 352,419 |

The opening text establishes their document identities and contains readable
prose and report sections. The conference proceedings identify the May 2006 event
and 2007 ECB publication; the catalog is not a substitute for those source dates.
[Sample evidence](ecb-other-publications-20261018.json) records original URLs,
retrieval timestamps, raw hashes, extracted hashes and opening text. Samples
were selected from locally cached FOEDB rows, then fetched live with robots
checks and five-second pacing.

The catalog count is not a saved-document or verified-token count. Translations
can expand candidate URLs, while exact-content duplicates can reduce saved rows.
Sample selection does not support an average-yield estimate or prove historical
retrievability. Unsupported formats remain excluded by the shared representation
rules. Passage overlap with articles and reports is not measured.

Validation: 288 tests passed, including the shared exact-type, native HTML
preference and distinct translation representation case for this addition.
Ruff lint/format and whitespace checks passed. Existing source campaigns and
reservations remain in charge of dispatch; deployment does not start another
source writer or reset any failures. Full inventory comparison and subsequent
worker adoption are recorded separately in the continuation report.

Full enumeration subsequently produced **771 supported candidate URLs** from
772 catalog publication records. There is zero exact URL overlap with the 21
prior source collections (14,372 unique URLs at the pinned source revision),
and zero overlap with the four report-series additions. Two catalog records share the same
`whypricestabilityen.pdf` URL; every record has a supported representation. The [inventory evidence](ecb-other-inventory-overlap-20261018.json)
records enumeration completeness, receipts and the sorted-URL fingerprint.
This checks the chosen catalog scope; it does not establish whole-site coverage.
